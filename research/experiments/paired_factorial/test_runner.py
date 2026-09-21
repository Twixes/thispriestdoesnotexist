"""Small synthetic64 mechanics; no real pretrained checkpoint, images or D export loaded."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('factorial_runner',HERE/'runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class Static(unittest.TestCase):
    def test_guarded_entrypoint(self):
        command=f"import runpy,sys;runpy.run_path({str(HERE/'runner.py')!r},run_name='audit');assert 'torch' not in sys.modules and 'pickle' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'unused'
            result=subprocess.run([sys.executable,str(HERE/'runner.py'),'--output',str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(output.exists())
        self.assertEqual(runner.UPDATES,300);self.assertEqual(runner.MILESTONES,(0,100,200,300));self.assertEqual(runner.MAX_RSS,10*1024**3);self.assertEqual(runner.MIN_FREE_PERCENT,35)
        self.assertEqual(runner.MAX_SECONDS,14400)
        from unittest.mock import patch
        from types import SimpleNamespace
        with patch.object(runner.sys,'platform','darwin'):
            for free in [34,35]:
                with patch.object(runner.subprocess,'run',return_value=SimpleNamespace(stdout=f'System-wide memory free percentage: {free}%')):
                    if free==34:
                        with self.assertRaises(RuntimeError):runner.memory_guard()
                    else:self.assertIn('35%',runner.memory_guard())


class Tiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);cls.t=torch
        from research.experiments.paired_regions import trainer
        from research.experiments.paired_factorial.optimizer import fork_optimizer
        from research.experiments.paired_factorial.update import update, FEATURE_WEIGHT
        from research.experiments.clothing_structure.losses import FrozenDPrefix,feature_reconstruction_loss
        from research.experiments.paired_factorial import render
        from training.networks import Discriminator
        cls.tr=trainer;cls.fork=staticmethod(fork_optimizer);cls.update=staticmethod(update);cls.feature_weight=FEATURE_WEIGHT
        cls.Prefix=FrozenDPrefix;cls.feature=staticmethod(feature_reconstruction_loss);cls.D=Discriminator;cls.render=render

    def fixture(self):
        t=self.t;tr=self.tr;t.manual_seed(401)
        source=tr.Generator(z_dim=8,c_dim=0,w_dim=8,img_resolution=64,img_channels=3,
                            mapping_kwargs={'num_layers':2},synthesis_kwargs={'channel_base':256,'channel_max':16}).eval().requires_grad_(False)
        with t.no_grad():
            for n,p in source.named_parameters():
                if n.endswith('noise_strength'):p.fill_(.13)
        mask=tr.polygon_mask([[[0,.75],[1,.75],[1,1],[0,1]]],64)
        collar,rest,_=tr.partition_masks(mask,[[[.4,.8],[.55,.8],[.55,.9],[.4,.9]]],64)
        pairs=[];rng=t.Generator().manual_seed(76)
        for i in range(10):
            z=t.randn(1,8,generator=rng)
            with t.no_grad():w=source.mapping(z,None,truncation_psi=1,skip_w_avg_update=True);original=source.synthesis(w,noise_mode='const',force_fp32=True)
            target=original*(1-mask)+.9*collar+(-.7+i*.03)*rest
            split='train' if i<6 else ('validation' if i==6 else 'diagnostic-heldout')
            pairs.append({'id':str(i),'split':split,'z':z,'w':w,'original':original,'target':target,'mask':mask,'collar':collar if i!=6 else None,'rest':rest if i!=6 else None})
        student=copy.deepcopy(source);tr.freeze_student(student)
        opt=t.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999))
        options={'batch':1,'clothing_weight':1.,'protected_weight':1.,'fresh_weight':1.,'upper_fraction':.75}
        sampling=t.Generator().manual_seed(77);preservation=t.Generator().manual_seed(78)
        for _ in range(2):tr.update(student,source,opt,pairs[:6],options,sampling,preservation,'cpu')
        parent={'student':copy.deepcopy(student.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'rng':tr.capture_rng('cpu',sampling,preservation)}
        prefix=self.Prefix(self.D(c_dim=0,img_resolution=64,img_channels=3,channel_base=256,channel_max=16),64,(32,16))
        return source,pairs,options,parent,prefix

    def test_four_actual_arms_optimizer_freezing_and_balanced_schedule(self):
        t=self.t;tr=self.tr;source,pairs,options,parent,prefix=self.fixture()
        source_digest=tr.state_digest(source.state_dict());d_digest=tr.state_digest(prefix.state_dict());initial=tr.state_digest(parent['student']);schedules=[]
        for arm in runner.ARMS:
            student=copy.deepcopy(source);student.load_state_dict(parent['student'])
            optimizer,transfer=self.fork(student,parent['optimizer'],arm in ('B','D'),tr,t)
            self.assertEqual(tr.state_digest(student.state_dict()),initial)
            old=optimizer.state_dict();old={'state':{k:v for k,v in old['state'].items() if k in parent['optimizer']['state']},'param_groups':old['param_groups'][:1]}
            self.assertTrue(runner.equal_tree(old,parent['optimizer'],t));self.assertTrue(transfer['old_optimizer_state_exact'])
            if arm in ('B','D'):
                self.assertTrue(transfer['new_parameter_names']);self.assertTrue(all(p not in optimizer.state for p in optimizer.param_groups[1]['params']))
            else:self.assertEqual(transfer['new_parameter_names'],[])
            self.assertFalse(any(p.requires_grad for n,p in student.named_parameters() if n.startswith(('mapping.','synthesis.b4.','synthesis.b8.','synthesis.b16.')) or n.endswith('noise_strength')))
            frozen=runner.shared.invariant_state(student,source,tr)
            s=t.Generator();p=t.Generator();tr.restore_rng(parent['rng'],'cpu',s,p)
            plan=runner.balanced_plan(s.get_state(),6,t)
            self.assertEqual(len(plan),300);self.assertEqual([plan.count(i) for i in range(6)],[50]*6)
            draws=[]
            for i in range(3):
                draw,expected=runner.planned_draw(i,plan,s,p,pairs[:6],8,t)
                metrics=self.update(student,source,optimizer,pairs[draw['pair_index']],options,p,tr,prefix if arm in ('C','D') else None)
                self.assertTrue(t.equal(p.get_state(),expected));self.assertEqual(metrics['pair_ids'],[draw['pair_id']]);draws.append(draw)
            self.assertNotEqual(tr.state_digest(student.state_dict()),initial)
            self.assertEqual(runner.shared.invariant_state(student,source,tr),frozen)
            self.assertEqual(tr.state_digest(source.state_dict()),source_digest);self.assertEqual(tr.state_digest(prefix.state_dict()),d_digest)
            self.assertFalse(any(p.grad is not None for p in source.parameters()));self.assertFalse(any(p.grad is not None for p in prefix.parameters()))
            if arm in ('B','D'):
                self.assertTrue(all(p in optimizer.state for p in optimizer.param_groups[1]['params']))
                self.assertTrue(any(not t.equal(parent['student'][n],v) for n,v in student.state_dict().items() if n.startswith('synthesis.b32.')))
            schedules.append(draws)
            with self.assertRaises(ValueError):self.update(student,source,optimizer,pairs[-1],options,p,tr,prefix)
        self.assertTrue(all(s==schedules[0] for s in schedules))

    def test_diagnostic_ordinary_equality_and_checkpoint_preview(self):
        t=self.t;tr=self.tr;source,pairs,options,parent,prefix=self.fixture();replicas=[]
        for diagnostic in [True,False]:
            student=copy.deepcopy(source);student.load_state_dict(parent['student'])
            optimizer,_=self.fork(student,parent['optimizer'],True,tr,t)
            s=t.Generator();p=t.Generator();tr.restore_rng(parent['rng'],'cpu',s,p)
            phases=[]
            metrics=self.update(student,source,optimizer,pairs[0],options,p,tr,prefix,diagnostic,phases.append if diagnostic else None)
            if diagnostic:
                self.assertEqual(phases,['before_feature_forward','after_feature_forward','before_diagnostic_pixel_autograd','after_diagnostic_pixel_autograd','before_diagnostic_feature_autograd','after_diagnostic_feature_autograd','before_paired_pixel_plus_feature_backward','after_paired_pixel_plus_feature_backward','before_fresh_preservation_backward','after_fresh_preservation_backward'])
            if diagnostic:
                cal=metrics['gradient_calibration']
                for key in ['existing64plus','new_b32']:
                    self.assertGreater(cal['pixel_l2'][key],0);self.assertGreater(cal['weighted_feature_l2'][key],0)
                    self.assertEqual(self.feature_weight,1.0)
                    self.assertEqual(metrics['feature_weight'],1.0)
                    self.assertEqual(cal['feature'],'weighted 1.0 frozen-D term')
                    self.assertEqual(cal['raw_feature_l2'][key],cal['weighted_feature_l2'][key]/self.feature_weight)
            replicas.append({'student':copy.deepcopy(student.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':tr.capture_rng('cpu',s,p)})
        self.assertTrue(runner.equal_tree(replicas[0],replicas[1],t))
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);fixed=t.randn(4,8,generator=t.Generator().manual_seed(999))
            report=self.render.snapshot(student,pairs,fixed,directory,0,tr,t,s,p)
            self.assertEqual(report['count'],14);self.assertTrue(runner.equal_tree(replicas[-1]['rng'],tr.capture_rng('cpu',s,p),t))
            from PIL import Image
            with Image.open(directory/'preview-000'/'0-gray.png') as image:self.assertEqual(image.mode,'L')
            with Image.open(directory/'preview-000'/'contact.png') as image:self.assertEqual(image.size,(1024,1120))
            tr.atomic_save(replicas[-1],directory/'tiny.pt');loaded=t.load(directory/'tiny.pt',weights_only=True,mmap=True)
            self.assertTrue(runner.equal_tree(loaded,replicas[-1],t))

    def test_feature_off_equals_original_fixed_pair_update(self):
        t=self.t;tr=self.tr;source,pairs,options,parent,prefix=self.fixture();results=[]
        for original in [True,False]:
            student=copy.deepcopy(source);student.load_state_dict(parent['student'])
            optimizer,_=self.fork(student,parent['optimizer'],False,tr,t)
            sampling=t.Generator();preservation=t.Generator();tr.restore_rng(parent['rng'],'cpu',sampling,preservation)
            if original:tr.update(student,source,optimizer,[pairs[2]],options,sampling,preservation,'cpu')
            else:self.update(student,source,optimizer,pairs[2],options,preservation,tr)
            rng=tr.capture_rng('cpu',sampling,preservation);del rng['sampling']
            results.append({'student':copy.deepcopy(student.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'rng_except_changed_sampling_policy':rng})
        self.assertTrue(runner.equal_tree(results[0],results[1],t))

    def test_saved_source30_schema_and_hashes_without_model(self):
        import hashlib
        import numpy as np
        from PIL import Image
        pins=json.loads((HERE/'parent-pins.json').read_text());path=runner.ROOT/pins['source30_manifest']['path']
        self.assertEqual(runner.sha(path),pins['source30_manifest']['sha256']);data=json.loads(path.read_text())
        self.assertEqual([p['id'] for p in data['entries']],[f'{i:03}' for i in range(32)])
        self.assertEqual(data['model']['model_sha256'],pins['provenance']['source_weights_sha256'])
        rows=[];excluded=[]
        train_z=set()
        for pair in pins['provenance']['pairs']:
            if pair['split']!='train':continue
            with np.load(pair['files']['latent_path']['path'],allow_pickle=False) as z:
                train_z.add(hashlib.sha256(z['z'].tobytes()).hexdigest())
        for entry in data['entries']:
            latent=path.parent/entry['latent_path'];source=path.parent/entry['source_path']
            self.assertEqual(runner.sha(latent),entry['latents_sha256']);self.assertEqual(runner.sha(source),entry['image_sha256'])
            with np.load(latent,allow_pickle=False) as value:
                self.assertEqual(value['z'].shape,(1,512));self.assertEqual(value['w'].shape,(1,18,512))
                self.assertEqual(value['z'].dtype,np.float32);self.assertEqual(value['w'].dtype,np.float32)
                self.assertTrue(np.isfinite(value['z']).all());self.assertTrue(np.isfinite(value['w']).all())
                zh=hashlib.sha256(value['z'].tobytes()).hexdigest()
                self.assertEqual(zh,entry['z_raw_sha256']);self.assertEqual(hashlib.sha256(value['w'].tobytes()).hexdigest(),entry['ws_raw_sha256'])
            with Image.open(source) as image:self.assertEqual(image.size,(1024,1024));self.assertEqual(image.mode,'RGB')
            (excluded if zh in train_z else rows).append(entry['id'])
        self.assertEqual(excluded,['000','030']);self.assertEqual(len(rows),30)
        (HERE/'source30-preflight.json').write_text(json.dumps({'manifest_sha256':runner.sha(path),'entry_count':32,'heldout_count':30,'excluded_training_ids':excluded,'heldout_ids':rows,'all_file_hashes_verified':True,'all_latent_shapes_dtypes_raw_hashes_finite':True,'all_png_rgb1024_headers':True,'model_inference_performed':False},indent=2)+'\n')

    def test_feature_input_gradient_is_confined_to_clothing(self):
        t=self.t;source,pairs,options,parent,prefix=self.fixture();pair=pairs[0]
        rgb=(pair['original']+.1).detach().requires_grad_(True)
        loss,_=self.feature(prefix,rgb,pair['target'],pair['original'],pair['mask'],pair['collar'])
        gradient=t.autograd.grad(loss,rgb)[0]
        self.assertEqual(float((gradient*(1-pair['mask'])).abs().sum()),0.)
        self.assertGreater(float((gradient*pair['mask']).abs().sum()),0.)
        self.assertTrue(bool(t.isfinite(gradient).all()))


if __name__=='__main__':unittest.main(verbosity=2)
