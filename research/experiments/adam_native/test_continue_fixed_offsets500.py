"""Tiny exact-next-update continuation tests; no pretrained/native forward loads."""
import ast
import copy
import io
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F
from continue_output_rank1_fixed_offsets500 import (
    restore_training_state, expected_optimizer_steps, START_ITERATION, ITERATIONS,
    CHECKPOINT_STEPS, PREVIEW_STEPS,
)

HERE=Path(__file__).resolve().parent


def tiny_state():
    g=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh(),torch.nn.Linear(4,3))
    d=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Softplus(),torch.nn.Linear(4,1))
    models={'G':g,'D':d,'Gema':copy.deepcopy(g).requires_grad_(False),'Dema':copy.deepcopy(d).requires_grad_(False)}
    optimizers={'g_optimizer':torch.optim.Adam(g.parameters(),lr=.0016,betas=(0.,.99**(.8))),
                'd_optimizer':torch.optim.Adam(d.parameters(),lr=.002*16/17,betas=(0.,.99**(16/17)))}
    return models,optimizers,torch.Generator().manual_seed(92)


def tiny_iteration(models,optimizers,rng,path_mean,counts,iteration):
    g,d=models['G'],models['D'];go,do=optimizers['g_optimizer'],optimizers['d_optimizer']
    def update(opt,loss,key):
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();counts[key]+=1
    def fake():
        # Separate CPU sampling and global synthesis-noise streams both matter.
        return g(torch.randn(2,3,generator=rng))+torch.randn(2,3)*.01
    real=torch.randn(2,3,generator=rng)
    update(do,F.softplus(d(fake().detach())).mean()+F.softplus(-d(real)).mean(),'d')
    if iteration%16==0:
        x=real.detach().requires_grad_(True)
        grad,=torch.autograd.grad(d(x).sum(),x,create_graph=True)
        update(do,grad.square().sum()+d(x).sum()*0,'d')
    update(go,F.softplus(-d(fake())).mean(),'g')
    if iteration%4==0:
        # Stateful scalar moving target, independent G optimizer update.
        score=fake().square().mean()
        target=path_mean+.01*(score-path_mean)
        update(go,(score-target).square(),'g')
        path_mean=target.detach()
    with torch.no_grad():
        for en,n in [('Gema','G'),('Dema','D')]:
            for pe,p in zip(models[en].parameters(),models[n].parameters()):pe.copy_(p.lerp(pe,.95))
    return path_mean


def snapshot(models,optimizers,rng,path_mean,counts,iteration):
    return {**{n:m.state_dict() for n,m in models.items()},**{n:o.state_dict() for n,o in optimizers.items()},
            'torch_rng':torch.get_rng_state(),'sampling_rng':rng.get_state(),'path_mean':path_mean,
            'optimizer_steps':dict(counts),'iterations':iteration}


def clone_serialized(value):
    buffer=io.BytesIO();torch.save(value,buffer);buffer.seek(0)
    return torch.load(buffer,weights_only=True)


def exact(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b));return
    if isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a:exact(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):exact(test,x,y)
    else:test.assertEqual(a,b)


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.set_num_interop_threads(1)

    def test_restore_then_one_next_update_equals_uninterrupted(self):
        torch.manual_seed(81);models,optimizers,rng=tiny_state();counts={'g':0,'d':0};path=torch.zeros(())
        for iteration in range(100):path=tiny_iteration(models,optimizers,rng,path,counts,iteration)
        self.assertEqual(counts,{'g':125,'d':107})
        saved=clone_serialized(snapshot(models,optimizers,rng,path,counts,100))
        expected_path=tiny_iteration(models,optimizers,rng,path,counts,100)
        expected=clone_serialized(snapshot(models,optimizers,rng,expected_path,counts,101))
        # Construct unrelated models/optimizers, and disturb both RNG streams.
        restored,restored_opt,restored_rng=tiny_state();torch.randn(11);torch.randn(9,generator=restored_rng)
        restored_path,restored_counts=restore_training_state(saved,restored,restored_opt,restored_rng,expected_iteration=100)
        exact(self,saved,snapshot(restored,restored_opt,restored_rng,restored_path,restored_counts,100))
        next_path=tiny_iteration(restored,restored_opt,restored_rng,restored_path,restored_counts,100)
        actual=snapshot(restored,restored_opt,restored_rng,next_path,restored_counts,101)
        exact(self,expected,actual)
        self.assertEqual(restored_counts,{'g':127,'d':108})
        # Prove this test is sensitive to the optimizer and moving-target state.
        stale,stale_opt,stale_rng=tiny_state()
        stale_path,stale_counts=restore_training_state(clone_serialized(saved),stale,stale_opt,stale_rng,expected_iteration=100)
        stale_opt['g_optimizer'].state.clear()
        stale_path=tiny_iteration(stale,stale_opt,stale_rng,stale_path+1,stale_counts,100)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(stale['G'].parameters(),restored['G'].parameters())))

    def test_counts_rejected_before_model_mutation(self):
        models,optimizers,rng=tiny_state();saved=clone_serialized(snapshot(models,optimizers,rng,torch.zeros(()),{'g':125,'d':107},100))
        before=clone_serialized(models['G'].state_dict())
        saved['optimizer_steps']['g']=124
        with self.assertRaises(ValueError):restore_training_state(saved,models,optimizers,rng,expected_iteration=100)
        exact(self,before,models['G'].state_dict())

    def test_schedule_and_loss_loop_are_unchanged(self):
        self.assertEqual((START_ITERATION,ITERATIONS),(100,500))
        self.assertEqual(CHECKPOINT_STEPS,(250,500));self.assertEqual(PREVIEW_STEPS,(100,250,500))
        self.assertEqual(expected_optimizer_steps(250),{'g':313,'d':266})
        self.assertEqual(expected_optimizer_steps(500),{'g':625,'d':532})
        def worker(filename):
            return next(n for n in ast.parse((HERE/filename).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        parent=worker('probe_output_rank1_fixed_offsets100.py');child=worker('continue_output_rank1_fixed_offsets500.py')
        for name in ('generate','update','apply_step','gradient_metrics'):
            functions=lambda w:next(n for n in w.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(functions(parent)),ast.dump(functions(child)))
        loop=lambda w:next(n for n in w.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
        self.assertEqual(ast.dump(ast.Module(body=loop(parent).body,type_ignores=[])),ast.dump(ast.Module(body=loop(child).body,type_ignores=[])))
        self.assertEqual(ast.dump(loop(child).iter),"Call(func=Name(id='range', ctx=Load()), args=[Name(id='START_ITERATION', ctx=Load()), Name(id='ITERATIONS', ctx=Load())], keywords=[])")


if __name__=='__main__':unittest.main(verbosity=2)
