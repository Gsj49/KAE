"""Check RNG replay against the historical functions, not an approximation."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/gradient_mse'))
from global_plan import legacy_draws,batch_totals


def historical_functions(filename):
    tree=ast.parse(filename.read_text())
    names={'choose_prompt_bundles','build_sampling_plan','compute_reinforcepp_batch_mean_baselines'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    stripped=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    namespace={'np':np,'PromptBundle':SimpleNamespace,
        'compute_mean_var':lambda r:(float(np.mean(r)),float(np.var(r)))}
    exec(compile(ast.fix_missing_locations(stripped),str(filename),'exec'),namespace)
    return namespace


class GlobalPlanTest(unittest.TestCase):
    def test_exact_legacy_draws_and_shard_invariant_batch(self):
        uids=sorted(str(i) for i in range(256))
        steps=[6,7,8,9,10]
        rng=np.random.default_rng(90)
        pools={s:{uid:SimpleNamespace(messages=[{'role':'user','content':uid}],responses=['x']*128,
            rewards=rng.integers(0,2,128).astype(np.float32)) for uid in uids} for s in steps}
        order,draws=legacy_draws(uids,np.full((256,5),128))
        old=historical_functions(ROOT/'experiments/gradient_mse/gsm8k.py')
        for shard in range(16):
            random=np.random.default_rng(0)
            bundles,_=old['choose_prompt_bundles'](pools,steps,10,8,0,random,True,16,shard)
            plan=old['build_sampling_plan'](bundles,steps,8,50,random,True)
            for local_index,bundle in enumerate(bundles):
                global_index=local_index*16+shard
                self.assertEqual(order[global_index],bundle.uid)
                for repeat in range(50):
                    for j,step in enumerate(steps):
                        np.testing.assert_array_equal(draws[global_index,repeat,j],plan[local_index][repeat][step])
        totals,count=batch_totals(order,draws,{u:s.rewards for u,s in pools[10].items()})
        self.assertEqual(count,2048)
        for repeat in (0,17,49):
            values=np.concatenate([pools[10][u].rewards[draws[i,repeat,-1]] for i,u in enumerate(order)])
            self.assertEqual(totals[repeat],values.sum())
            self.assertAlmostEqual((totals[repeat]-values[0])/(count-1),np.delete(values,0).astype(float).mean())
        for name in ('choose_prompt_bundles','build_sampling_plan'):
            a=next(n for n in ast.parse((ROOT/'experiments/gradient_mse/gsm8k.py').read_text()).body if getattr(n,'name',None)==name)
            b=next(n for n in ast.parse((ROOT/'experiments/gradient_mse/math_mse.py').read_text()).body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(a),ast.dump(b))

    def test_gpu_scalar_reduction_algebra_on_cpu(self):
        import torch
        from global_loo import scalar_squared_error
        params=[torch.nn.Parameter(torch.zeros(10,dtype=torch.bfloat16)),torch.nn.Parameter(torch.zeros(3,4,dtype=torch.bfloat16))]
        reference=[torch.randn_like(p) for p in params]
        old=[torch.zeros_like(p,dtype=torch.float32) for p in params]
        new=0.
        for _ in range(50):
            for p in params:
                p.grad=torch.randn_like(p)
            new+=scalar_squared_error(params,reference).item()
            for j,p in enumerate(params):
                old[j]+=(p.grad.float()-reference[j].float()).square()
        old_value=sum((v/50).sum().item() for v in old)
        self.assertAlmostEqual(new/50,old_value,delta=old_value*1e-6)


if __name__=='__main__':
    unittest.main()
