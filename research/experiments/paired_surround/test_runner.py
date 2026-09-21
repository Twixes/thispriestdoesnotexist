"""Small synthetic64px mechanics only; no pretrained files or1024 model loads."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('surround_runner',HERE/'runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class StaticGuards(unittest.TestCase):
    def test_no_implicit_real_execution_or_model_import(self):
        code=f"import runpy,sys;runpy.run_path({str(HERE/'runner.py')!r},run_name='audit');assert 'torch' not in sys.modules and 'pickle' not in sys.modules"
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'unused'
            r=subprocess.run([sys.executable,str(HERE/'runner.py'),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertFalse(out.exists())
        self.assertEqual(runner.UPDATES,50);self.assertEqual(runner.MILESTONES,(0,25,50))
        self.assertEqual(runner.BRANCHES,('baseline','collar-surround'))


class TinyMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        cls.t=torch
        sys.path.insert(0,str(runner.ROOT))
        from research.experiments.paired_regions import trainer
        from research.experiments.paired_surround import objective
        cls.tr=trainer;cls.obj=objective

    def test_partition_chebyshev_normalization_and_no_protected_gradient(self):
        t=self.t
        mask=t.zeros(1,1,16,16);mask[:,:,7:,:]=1
        raw=t.zeros_like(mask);raw[:,:,8:10,3:5]=1
        c,s,r=self.obj.partition(mask,raw,radius=2)
        expected=t.zeros_like(mask);expected[:,:,6:12,1:7]=1
        expected=expected*(1-raw)*mask
        self.assertTrue(t.equal(s,expected));self.assertTrue(t.equal(c+s+r,mask))
        self.assertFalse(bool((s*(1-mask)).any()))
        generated=t.ones_like(mask,requires_grad=True)
        target=t.zeros_like(mask)
        loss,*components=self.obj.clothing_loss(generated,target,c,s,r,self.tr.masked_l1)
        self.assertEqual(float(loss.detach()),1.0);self.assertTrue(all(float(v.detach())==1 for v in components))
        loss.backward()
        self.assertEqual(float((generated.grad*(1-mask)).abs().sum()),0)
        for region in [c,s,r]:self.assertAlmostEqual(float((generated.grad*region).sum()),1/3,places=6)
        with self.assertRaises(ValueError):self.obj.partition(mask,t.zeros_like(mask),radius=2)
        with self.assertRaises(ValueError):self.obj.partition(mask,mask,radius=2)

    def fixture(self):
        t=self.t;tr=self.tr;t.manual_seed(401)
        source=tr.Generator(z_dim=8,c_dim=0,w_dim=8,img_resolution=64,img_channels=3,
                            mapping_kwargs={'num_layers':2},synthesis_kwargs={'channel_base':256,'channel_max':16}).eval().requires_grad_(False)
        with t.no_grad():
            for n,p in source.named_parameters():
                if n.endswith('noise_strength'):p.fill_(.13)
        mask=tr.polygon_mask([[[0,.75],[1,.75],[1,1],[0,1]]],64)
        polygons=[[[.04,.84],[.13,.84],[.13,.94],[.04,.94]]]
        collar,rest,_=tr.partition_masks(mask,polygons,64)
        pairs=[];manifest={'pairs':[]};rng=t.Generator().manual_seed(76)
        for i in range(7):
            z=t.randn(1,8,generator=rng)
            with t.no_grad():w=source.mapping(z,None,truncation_psi=1,skip_w_avg_update=True);original=source.synthesis(w,noise_mode='const',force_fp32=True)
            target=original*(1-mask)+.9*collar+(-.7+i*.03)*rest
            split='train' if i<6 else 'validation';ident=str(i) if i<6 else '028'
            pairs.append({'id':ident,'split':split,'z':z,'w':w,'original':original,'target':target,'mask':mask,'collar':collar if i<6 else None,'rest':rest if i<6 else None})
            manifest['pairs'].append({'id':ident,'split':split,'collar_polygons':polygons})
        self.obj.attach_regions(pairs,manifest,tr)
        student=copy.deepcopy(source);tr.freeze_student(student)
        optimizer=t.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999))
        options={'batch':1,'clothing_weight':1.,'protected_weight':1.,'fresh_weight':1.,'upper_fraction':.75}
        sampling=t.Generator().manual_seed(77);preservation=t.Generator().manual_seed(78)
        for _ in range(2):tr.update(student,source,optimizer,pairs[:6],options,sampling,preservation,'cpu')
        parent={'student':copy.deepcopy(student.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':tr.capture_rng('cpu',sampling,preservation)}
        return source,pairs,options,parent

    def test_actual_updates_matched_rng_optimizer_and_invariants(self):
        t=self.t;tr=self.tr
        source,pairs,options,parent=self.fixture();parent_hash=tr.state_digest(parent['student'])
        source_hash=tr.state_digest(source.state_dict());schedules=[];initial=[]
        fixed=t.randn(4,8,generator=t.Generator().manual_seed(800))
        for branch in runner.BRANCHES:
            student=copy.deepcopy(source);student.load_state_dict(parent['student'])
            optimizer,info=runner.restore_optimizer(student,parent['optimizer'],tr,t)
            self.assertTrue(runner.equal_tree(optimizer.state_dict(),parent['optimizer'],t));self.assertFalse(info['new_parameters'])
            self.assertFalse(any(p.requires_grad for n,p in student.named_parameters() if n.startswith('mapping.') or n.startswith(('synthesis.b4.','synthesis.b8.','synthesis.b16.','synthesis.b32.')) or n.endswith('noise_strength')))
            self.assertEqual(tr.state_digest(student.state_dict()),parent_hash);initial.append(parent_hash)
            invariants=runner.invariant_state(student,source,tr)
            s=t.Generator();p=t.Generator();tr.restore_rng(parent['rng'],'cpu',s,p)
            draws=[]
            with tempfile.TemporaryDirectory() as temp:
                output=Path(temp)
                rng_before=tr.capture_rng('cpu',s,p)
                report=runner.snapshot(student,pairs,fixed,output,0,tr,t,s,p)
                self.assertEqual(report['count'],11);self.assertTrue(runner.equal_tree(rng_before,tr.capture_rng('cpu',s,p),t))
                for i in range(3):
                    draw,se,pe=runner.predict_draws(s,p,pairs[:6],options,source.z_dim,t)
                    if branch=='baseline':metrics=tr.update(student,source,optimizer,pairs[:6],options,s,p,'cpu')
                    else:metrics=self.obj.update(student,source,optimizer,pairs[:6],options,s,p,tr)
                    self.assertEqual(metrics['pair_ids'],draw['pair_ids']);self.assertTrue(t.equal(s.get_state(),se));self.assertTrue(t.equal(p.get_state(),pe));draws.append(draw)
                    self.assertTrue(all(__import__('math').isfinite(v) for v in metrics.values() if isinstance(v,float)))
                self.assertEqual(runner.invariant_state(student,source,tr),invariants)
                self.assertNotEqual(tr.state_digest(student.state_dict()),parent_hash)
                self.assertEqual(tr.state_digest(source.state_dict()),source_hash)
                self.assertFalse(any(p.grad is not None for p in source.parameters()))
                checkpoint={'format':'paired-surround-fork-v1','student':student.state_dict(),'optimizer':optimizer.state_dict(),'rng':tr.capture_rng('cpu',s,p)}
                tr.atomic_save(checkpoint,output/'tiny.pt');loaded=t.load(output/'tiny.pt',weights_only=True,mmap=True)
                self.assertTrue(runner.equal_tree(loaded,checkpoint,t))
            schedules.append(draws)
        self.assertEqual(initial[0],initial[1]);self.assertEqual(schedules[0],schedules[1])
        # Forecast all50 draws independently from the same parent RNG, no updates.
        full=[]
        for _ in runner.BRANCHES:
            s=t.Generator();p=t.Generator();s.set_state(parent['rng']['sampling']);p.set_state(parent['rng']['preservation']);rows=[]
            for step in range(50):
                draw,se,pe=runner.predict_draws(s,p,pairs[:6],options,8,t);s.set_state(se);p.set_state(pe);rows.append(draw)
            full.append(rows)
        self.assertEqual(full[0],full[1])

    def test_all_six_real_saved_mask_partitions_only(self):
        # JSON polygons only; no images/NPZ/checkpoints/source-model weights loaded.
        manifest=json.loads((runner.ROOT/'research/data/paired7/manifest-proposed-v2.json').read_text())
        rows=[]
        for row in manifest['pairs']:
            if row['split']!='train':continue
            mask=self.tr.polygon_mask(row['clothing_polygons'],1024);raw=self.tr.polygon_mask(row['collar_polygons'],1024)
            c,s,r=self.obj.partition(mask,raw)
            self.assertTrue(self.t.equal(c+s+r,mask))
            rows.append({'id':row['id'],'clothing_pixels':int(mask.sum()),'collar_pixels':int(c.sum()),'surround_pixels':int(s.sum()),'remainder_pixels':int(r.sum()),'protected_leak_pixels':0})
        self.assertEqual(len(rows),6)
        (HERE/'mask-counts.json').write_text(json.dumps({'manifest_sha256':runner.sha(runner.ROOT/'research/data/paired7/manifest-proposed-v2.json'),'radius_pixels':30,'metric':'Chebyshev dilation of raw trace, excluding raw trace, intersect original clothing','rows':rows},indent=2)+'\n')


if __name__=='__main__':unittest.main(verbosity=2)
