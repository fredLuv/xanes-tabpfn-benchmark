"""Run the frozen benchmark sequentially with the selected model environment.

Usage: .venv-rf/bin/python scripts/reproduce.py --model rf --phase test
       .venv-tabpfn/bin/python scripts/reproduce.py --model tabpfn --phase importance --checkpoint weights/tabpfn-v3.5-20260909.safetensors

Existing successful cases are compared, not overwritten. Failed test cases are
reported and preserved. Importance checkpoints resume only with identical inputs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from xanes_benchmark.inference import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=['rf','tabpfn'])
    parser.add_argument('--phase',required=True,choices=['test','importance'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--data-dir',default=str(ROOT/'data'))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--runs-dir',default=str(ROOT/'runs'))
    args=parser.parse_args()
    if args.model=='tabpfn' and not args.checkpoint and not args.dry_run:
        parser.error('--checkpoint is required for TabPFN')
    tasks=['coord','bader','md'] if args.phase=='test' else ['coord']
    seeds=[42,43,44] if args.phase=='test' else [42]
    summary=[]
    for element in ['Ti','V','Cr','Mn','Fe','Co','Ni','Cu']:
        for task in tasks:
            for seed in seeds:
                name=f'{element}_{task}_{args.model}_{seed}' if args.phase=='test' else f'importance_{element}_{args.model}_42'
                folder=Path(args.runs_dir).resolve()/name
                command=[sys.executable,'-m','xanes_benchmark','run' if args.phase=='test' else 'importance',
                         '--model',args.model,'--element',element,'--task',task,'--seed',str(seed),'--data-dir',args.data_dir,'--runs-dir',str(Path(args.runs_dir).resolve())]
                if args.checkpoint: command+=['--checkpoint',str(Path(args.checkpoint).resolve())]
                if args.phase=='importance': command+=['--resume']
                if args.dry_run:
                    print(' '.join(command));continue
                if not (folder/'result.json').exists():
                    if args.phase=='test' and folder.exists():
                        summary.append(dict(case=name,status='existing_failed_or_incomplete',detail='Preserved; inspect execution.log.'));continue
                    completed=subprocess.run(command,cwd=ROOT)
                    if completed.returncode:
                        summary.append(dict(case=name,status='failed_or_limited',returncode=completed.returncode));continue
                try: summary.append(dict(case=name,status='complete',comparison=compare(folder)))
                except Exception as error: summary.append(dict(case=name,status='comparison_failed',error=str(error)))
    if args.dry_run:return
    destination=Path(args.runs_dir)/f'{args.model}_{args.phase}_summary.json'
    destination.write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Completed {sum(r["status"]=="complete" for r in summary)}/{len(summary)}. Summary: {destination}')
    if any(r['status']!='complete' for r in summary):raise SystemExit(1)


if __name__=='__main__':main()
