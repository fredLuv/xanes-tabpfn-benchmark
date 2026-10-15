"""Reproduce the published benchmark: inputs, model predictions, importance and figures."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '4'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/benchmark.json'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def prepare(args):
    import urllib.request
    import zipfile
    config = json.loads(CONFIG.read_text())
    directory = Path(args.data_dir); directory.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive) if args.archive else directory / 'xanes_2019.zip'
    if not archive.exists():
        if args.archive: raise FileNotFoundError(archive)
        part = archive.with_suffix('.partial')
        print('Downloading original archive (about 1.08 GB).', flush=True)
        urllib.request.urlretrieve(config['archive']['url'], part)
        if sha256(part) != config['archive']['sha256']: raise ValueError('Archive checksum mismatch')
        part.replace(archive)
    if sha256(archive) != config['archive']['sha256']: raise ValueError('Archive checksum mismatch')
    required = {a['filename']: a for c in config['cases'] for a in c['arrays'].values()}
    import numpy as np
    with zipfile.ZipFile(archive) as z:
        for name, spec in required.items():
            matches = [p for p in z.namelist() if p.endswith('/model_data/' + name)]
            if len(matches) != 1: raise ValueError(f'Expected one archived array: {name}')
            content = z.read(matches[0])
            if hashlib.sha256(content).hexdigest() != spec['sha256']: raise ValueError(name)
            out = directory / name; out.write_bytes(content)
            a = np.load(out, allow_pickle=False)
            if list(a.shape) != spec['shape'] or not np.isfinite(a).all(): raise ValueError(name)
    print('Verified and extracted 144 pointwise arrays; original row order preserved.')


def worker(args):
    import numpy as np
    from .inference import load_inputs, build_model, package_versions, importance_worker
    config, case, data = load_inputs(args)
    model = build_model(args, case, config)
    if args.kind == 'importance':
        return importance_worker(args, model, data)
    start = time.monotonic()
    model.fit(data['train_x'], data['train_y'])
    fit_seconds = time.monotonic() - start
    pred = np.asarray(model.predict(data['test_x'])).reshape(-1)
    if pred.shape != data['test_y'].shape or not np.isfinite(pred).all(): raise ValueError('Invalid predictions')
    out = Path(args.output)
    from .inference import score
    measured = score(data['test_y'], pred, args.task)
    np.savez_compressed(out / 'predictions.npz', y=data['test_y'], predictions=pred)
    (out / 'result.json').write_text(json.dumps(dict(status='success', model=args.model,
        element=args.element, task=args.task, seed=args.seed, metrics=measured, fit_seconds=fit_seconds,
        elapsed_seconds=time.monotonic()-start, config_sha256=sha256(CONFIG),
        runner_sha256=sha256(__file__), inference_sha256=sha256(ROOT/'xanes_benchmark/inference.py'),
        predictions_sha256=sha256(out/'predictions.npz'), packages=package_versions(args.model),
        n_estimators=getattr(model,'n_estimators_',getattr(model,'n_estimators',None))), indent=2)+'\n')


def run(args):
    import fcntl
    import subprocess
    import psutil
    folder = Path(args.runs_dir).resolve(); folder.mkdir(parents=True, exist_ok=True)
    (ROOT/'runs').mkdir(exist_ok=True)
    with open(ROOT/'runs/.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = folder / (f'importance_{args.element}_{args.model}_42' if args.kind == 'importance' else f'{args.element}_{args.task}_{args.model}_{args.seed}')
        if out.exists() and not (args.kind == 'importance' and args.resume):
            raise RuntimeError('Output already exists. Importance can resume with --resume; preserve test reruns separately.')
        out.mkdir(exist_ok=True)
        cmd = [sys.executable, '-m', 'xanes_benchmark', '_worker', '--element', args.element,
               '--task', args.task, '--model', args.model, '--seed', str(args.seed),
               '--data-dir', args.data_dir, '--output', str(out), '--kind', args.kind]
        if args.checkpoint: cmd += ['--checkpoint', args.checkpoint]
        if args.resume: cmd += ['--resume']
        start = time.monotonic(); peak = 0; reason = None
        with open(out / 'execution.log', 'a') as log:
            env = os.environ.copy()
            env.update(TABPFN_NO_BROWSER='1', TABPFN_MPS_MEMORY_FRACTION='0.6', DO_NOT_TRACK='1', HF_HUB_DISABLE_TELEMETRY='1')
            process = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=ROOT, env=env, start_new_session=True)
            try:
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True) if p.is_running())
                    except (psutil.NoSuchProcess, psutil.AccessDenied): rss = 0
                    try: gpu = int((out / 'mps_bytes.txt').read_text())
                    except (FileNotFoundError, ValueError): gpu = 0
                    peak = max(peak, rss + gpu)
                    if peak > 10_000_000_000: reason = 'memory'
                    if time.monotonic()-start > 1800: reason = 'time'
                    if reason:
                        import signal
                        os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=10); break
                    time.sleep(.2)
            finally:
                if process.poll() is None:
                    import signal
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
        (out / 'resources.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,
            peak_rss_plus_mps_bytes=peak, limit_reason=reason, returncode=process.returncode), indent=2)+'\n')
        print(out, 'complete' if process.returncode == 0 else f'failed/limited ({reason}); inspect execution.log')
        if process.returncode: raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['verify','report']: sub.add_parser(name)
    f = sub.add_parser('figures'); f.add_argument('--output-dir', default=str(ROOT/'output/figures'))
    c = sub.add_parser('compare'); c.add_argument('run_dir')
    d = sub.add_parser('prepare'); d.add_argument('--archive'); d.add_argument('--data-dir',default=str(ROOT/'data'))
    for name in ['run','importance','_worker']:
        q = sub.add_parser(name)
        q.add_argument('--model',choices=['rf','tabpfn'],required=True)
        q.add_argument('--element',choices=['Ti','V','Cr','Mn','Fe','Co','Ni','Cu'],required=True)
        q.add_argument('--task',choices=['coord','md','bader'],default='coord',required=name=='run')
        q.add_argument('--seed',type=int,choices=[42,43,44],default=42)
        q.add_argument('--data-dir',default=str(ROOT/'data')); q.add_argument('--checkpoint')
        q.add_argument('--runs-dir',default=str(ROOT/'runs'))
        q.add_argument('--resume', action='store_true', help='Resume verified importance checkpoints only')
        if name=='_worker':
            q.add_argument('--output',required=True)
            q.add_argument('--kind',choices=['test','importance'],required=True)
        else: q.set_defaults(kind='importance' if name=='importance' else 'test')
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command in ['run','importance']:
        if args.kind == 'importance' and (args.task != 'coord' or args.seed != 42):
            p.error('Released importance uses coordination, model seed 42 only')
        run(args)
    elif args.command=='_worker':worker(args)
    elif args.command=='figures':
        from .figures import build
        print(build(Path(args.output_dir)))
    elif args.command=='compare':
        from .inference import compare
        print(json.dumps(compare(Path(args.run_dir)), indent=2))
    else:
        from .analysis import verify, paired_summary
        print(verify())
        if args.command=='report':
            for task in ['coord','bader','md']:print(task+'\n'+paired_summary(task).round(5).to_string())

if __name__=='__main__':main()
