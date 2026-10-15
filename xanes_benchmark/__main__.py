"""Inspect saved results or rerun one frozen test case; no automatic experiment queue."""
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
    import importlib.metadata
    import numpy as np
    config = json.loads(CONFIG.read_text())
    case = next(c for c in config['cases'] if c['element'] == args.element and c['task'] == args.task)
    directory = Path(args.data_dir); data = {}
    for key, a in case['arrays'].items():
        f = directory / a['filename']
        if sha256(f) != a['sha256']: raise ValueError(f'Input checksum mismatch: {a["filename"]}')
        data[key] = np.load(f, allow_pickle=False)
        if list(data[key].shape) != a['shape'] or not np.isfinite(data[key]).all(): raise ValueError(key)
    version = importlib.metadata.version
    expected = {'rf': '1.3.2', 'tabpfn': '1.7.2'}[args.model]
    if version('scikit-learn') != expected: raise RuntimeError(f'Use pinned {args.model} environment: sklearn {expected}')
    if args.model == 'rf':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if args.task == 'coord' else RandomForestRegressor
        model = cls(n_estimators=300, max_depth=35, max_features=8, min_samples_leaf=1,
                    min_samples_split=2, n_jobs=4, random_state=args.seed)
    else:
        import torch
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        if version('tabpfn') != '9.1.0': raise RuntimeError('Use pinned TabPFN 9.1.0')
        if not args.checkpoint: raise ValueError('Supply --checkpoint; weights are not bundled or downloaded automatically.')
        if sha256(args.checkpoint) != config['checkpoint']['sha256']: raise ValueError('Checkpoint checksum mismatch')
        params = case['tabpfn_params'].copy()
        if params['device'] == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Frozen case uses Apple MPS; backend substitutions require a separately labeled experiment.')
        torch.set_num_threads(4)
        # A conservative RSS + MPS-driver accounting may double count shared memory.
        import threading
        def sample_mps():
            while True:
                b = torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0
                Path(args.output, 'mps_bytes.txt').write_text(str(b))
                time.sleep(.2)
        threading.Thread(target=sample_mps, daemon=True).start()
        cls = TabPFNClassifier if args.task == 'coord' else TabPFNRegressor
        model = cls.create_default_for_version(ModelVersion.V3_5, random_state=args.seed,
                  model_path=args.checkpoint, n_preprocessing_jobs=4, **params)
    start = time.monotonic()
    model.fit(data['train_x'], data['train_y'])
    fit_seconds = time.monotonic() - start
    pred = np.asarray(model.predict(data['test_x'])).reshape(-1)
    if pred.shape != data['test_y'].shape or not np.isfinite(pred).all(): raise ValueError('Invalid predictions')
    out = Path(args.output)
    np.savez_compressed(out / 'predictions.npz', y=data['test_y'], predictions=pred)
    (out / 'result.json').write_text(json.dumps(dict(status='success', model=args.model,
        element=args.element, task=args.task, seed=args.seed, fit_seconds=fit_seconds,
        elapsed_seconds=time.monotonic()-start, config_sha256=sha256(CONFIG),
        runner_sha256=sha256(__file__), packages={p:version(p) for p in ['numpy','scipy','scikit-learn']},
        n_estimators=getattr(model,'n_estimators_',getattr(model,'n_estimators',None))), indent=2)+'\n')


def run(args):
    import fcntl
    import subprocess
    import psutil
    folder = ROOT / 'runs'; folder.mkdir(exist_ok=True)
    with open(folder / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = folder / f'{args.element}_{args.task}_{args.model}_{args.seed}'
        if out.exists(): raise RuntimeError('Output already exists. Preserve it; choose a separate checkout for reruns.')
        out.mkdir()
        cmd = [sys.executable, '-m', 'xanes_benchmark', '_worker', '--element', args.element,
               '--task', args.task, '--model', args.model, '--seed', str(args.seed),
               '--data-dir', args.data_dir, '--output', str(out)]
        if args.checkpoint: cmd += ['--checkpoint', args.checkpoint]
        start = time.monotonic(); peak = 0; reason = None
        with open(out / 'execution.log', 'w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=ROOT, start_new_session=True)
            try:
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True) if p.is_running())
                    except psutil.NoSuchProcess: rss = 0
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
        print(out.relative_to(ROOT), 'complete' if process.returncode == 0 else f'failed/limited ({reason}); inspect execution.log')
        if process.returncode: raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['verify','report']: sub.add_parser(name)
    d = sub.add_parser('prepare'); d.add_argument('--archive'); d.add_argument('--data-dir',default=str(ROOT/'data'))
    for name in ['run','_worker']:
        q = sub.add_parser(name)
        q.add_argument('--model',choices=['rf','tabpfn'],required=True)
        q.add_argument('--element',choices=['Ti','V','Cr','Mn','Fe','Co','Ni','Cu'],required=True)
        q.add_argument('--task',choices=['coord','md','bader'],required=True)
        q.add_argument('--seed',type=int,choices=[42,43,44],default=42)
        q.add_argument('--data-dir',default=str(ROOT/'data')); q.add_argument('--checkpoint')
        if name=='_worker':q.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='run':run(args)
    elif args.command=='_worker':worker(args)
    else:
        from .analysis import verify, paired_summary
        print(verify())
        if args.command=='report':
            for task in ['coord','bader','md']:print(task+'\n'+paired_summary(task).round(5).to_string())

if __name__=='__main__':main()
