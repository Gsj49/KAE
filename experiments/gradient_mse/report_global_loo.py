"""Validate and aggregate the six saved-checkpoint global-LOO experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from global_plan import sha256, read_plan

METHODS={'reinforcepp':'REINFORCE++','grpo_no_norm':'GRPO','overt_no_norm':'KAE'}


def validate(directory,dataset,step,prompts=256,repeats=50):
    summary_path=directory/'summary.json'
    detail_path=directory/'prompt_details.parquet'
    summary=json.loads(summary_path.read_text())
    cfg=summary['config']
    required=dict(dataset=dataset,step=step,prompts=prompts,global_batch_prompts=256,
        global_batch_completions=2048,k_eval=8,m_repeats=repeats,seed=0,dtype='bf16',
        baseline='global_batch_completion_leave_one_out',
        grad_scope='all' if dataset=='gsm8k' else 'last5+model_norm+lm_head_no_embeddings',reference_pool_size=128,
        sampling_with_replacement=True,score_reduction='sum',cov_device='cuda',
        linear_alpha=0.2,variance_floor=0.01,history_steps=list(range(step-4,step)),
        reference_microbatch=16 if dataset=='gsm8k' else 8,repeat_microbatch=0)
    for key,value in required.items():
        if cfg.get(key)!=value:
            raise ValueError(f'{directory}: {key} expected {value!r}, got {cfg.get(key)!r}')
    meta,draws=read_plan(Path(__file__).resolve().parent/'draws'/dataset/f'step_{step}')
    if cfg.get('plan_sha256') != meta['plan_sha256']:
        raise ValueError('Result did not use the saved draw plan')
    if set(cfg['methods'])!=set(METHODS) or summary['status']!='complete':
        raise ValueError('Missing methods or incomplete result')
    if summary['num_prompts_succeeded']!=prompts or summary['num_prompts_failed']!=0:
        raise ValueError('Incorrect prompt counts')
    expected_parameters=1543714304 if dataset=='gsm8k' else 1710289920
    if summary['scope']['parameter_count']!=expected_parameters:
        raise ValueError('Incorrect paper parameter scope')
    if dataset=='math':
        scope=summary['scope']
        if (scope.get('selected_layer_ids') != [23,24,25,26,27] or scope.get('include_model_norm') is not True
                or scope.get('include_embed_tokens') is not False or scope.get('include_lm_head') is not True):
            raise ValueError('MATH must use the last five blocks, final model norm and output layer')
    if summary['detail_sha256']!=sha256(detail_path):
        raise ValueError('Detail hash mismatch')
    frame=pd.read_parquet(detail_path)
    if len(frame)!=prompts or frame.uid.nunique()!=prompts or set(frame.global_index)!=set(range(prompts)):
        raise ValueError('Invalid prompt coverage')
    if (frame.repeats!=repeats).any() or (frame.reference_pool_size!=128).any():
        raise ValueError('Incomplete repeats/reference pool')
    if not frame.plan_sha256.eq(meta['plan_sha256']).all():
        raise ValueError('Mixed per-prompt draw plans')
    ordered=frame.sort_values('global_index').uid.astype(str).tolist()
    if ordered != draws['uids'][:prompts].tolist():
        raise ValueError('Prompt identities do not match the fixed draw plan')
    for method in METHODS:
        col=f'trace_ref_error_{method}'
        if not np.isfinite(frame[col]).all() or (frame[col]<0).any():
            raise ValueError(f'Invalid {method} error')
        if not np.isclose(frame[col].mean(),summary['avg_trace_ref_error'][method],rtol=1e-12,atol=0):
            raise ValueError('Summary/detail mismatch')
    return summary,frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--campaign',type=Path,help='Original job-manifest directory')
    inputs.add_argument('--results-root',type=Path,help='Portable layout: <dataset>/step_<step>')
    args=parser.parse_args()
    root=args.campaign or args.results_root
    destination=root/'aggregate'
    if destination.exists():
        raise FileExistsError(destination)
    manifest=(json.loads((args.campaign/'formal_jobs.json').read_text()) if args.campaign else
              {'jobs':[dict(dataset=d,step=s,output=str(root/d/f'step_{s}'))
                       for d in ('gsm8k','math') for s in (10,90,170)]})
    expected={(dataset,step) for dataset in ('gsm8k','math') for step in (10,90,170)}
    if {(r['dataset'],r['step']) for r in manifest['jobs']}!=expected or len(manifest['jobs'])!=6:
        raise ValueError('Formal job manifest is not exactly 2 datasets x 3 steps')
    results,inputs=[],[]
    for record in manifest['jobs']:
        directory=Path(record['output'])
        log=Path(record['log']).read_text() if 'log' in record else None
        if log is not None and 'GLOBAL_LOO_DONE' not in log:
            raise ValueError(f'Missing completion marker: {record["job_id"]}')
        summary,frame=validate(directory,record['dataset'],record['step'])
        grpo=float(frame.trace_ref_error_grpo_no_norm.mean())
        kae=float(frame.trace_ref_error_overt_no_norm.mean())
        row=dict(dataset=record['dataset'],step=record['step'],prompts=256,
            parameter_count=summary['scope']['parameter_count'],kae_reduction_percent=100*(1-kae/grpo))
        for method,label in METHODS.items():
            values=frame[f'trace_ref_error_{method}']
            row[f'{label}_mean']=float(values.mean())
            row[f'{label}_sd']=float(values.std(ddof=1))
        results.append(row)
        inputs.append(dict(job_id=record.get('job_id'),summary=str(directory/'summary.json'),
            summary_sha256=sha256(directory/'summary.json'),detail_sha256=sha256(directory/'prompt_details.parquet'),
            log_sha256=sha256(Path(record['log'])) if 'log' in record else None))
    destination.mkdir()
    pd.DataFrame(results).to_csv(destination/'gradient_mse_summary.csv',index=False)
    (destination/'gradient_mse_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    receipt=dict(status='complete',runs=6,methods=list(METHODS.values()),inputs=inputs,
        output_sha256={p.name:sha256(p) for p in destination.iterdir()})
    (destination/'aggregate_complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('GLOBAL_LOO_AGGREGATE_DONE',json.dumps(results),flush=True)

if __name__=='__main__':
    main()
