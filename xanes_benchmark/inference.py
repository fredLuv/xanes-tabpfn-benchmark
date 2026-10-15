"""Fit frozen RF/TabPFN settings and recreate the released validation permutations.

No imports from the author's private workspace. No tuning, downloads of weights,
or test-data fitting. Baseline and shuffled predictions are compared explicitly.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(y, prediction, task):
    if task == 'coord':
        f1 = f1_score(y, prediction, labels=[4, 5, 6], average=None, zero_division=0)
        return dict(accuracy=float(accuracy_score(y, prediction)), macro_f1=float(f1.mean()),
                    **{f'f1_cn{c}': float(v) for c, v in zip([4, 5, 6], f1)})
    return dict(mae=float(mean_absolute_error(y, prediction)), r2=float(r2_score(y, prediction)))


def package_versions(model):
    names = ['numpy', 'scipy', 'scikit-learn'] + (['tabpfn', 'torch'] if model == 'tabpfn' else [])
    return {name: importlib.metadata.version(name) for name in names}


def load_inputs(args):
    config = json.loads((ROOT/'configs/benchmark.json').read_text())
    case = next(c for c in config['cases'] if c['element'] == args.element and c['task'] == args.task)
    data = {}
    for key, spec in case['arrays'].items():
        path = Path(args.data_dir)/spec['filename']
        if digest(path) != spec['sha256']: raise ValueError(f'Input hash mismatch: {spec["filename"]}')
        data[key] = np.load(path, allow_pickle=False)
        if list(data[key].shape) != spec['shape'] or not np.isfinite(data[key]).all(): raise ValueError(key)
    return config, case, data


def build_model(args, case, config):
    version = importlib.metadata.version
    expected = {'rf':'1.3.2', 'tabpfn':'1.7.2'}[args.model]
    if version('scikit-learn') != expected: raise RuntimeError(f'Use pinned {args.model} environment (sklearn {expected})')
    if args.model == 'rf':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if args.task == 'coord' else RandomForestRegressor
        return cls(n_estimators=300, max_depth=35, max_features=8, min_samples_leaf=1,
                   min_samples_split=2, n_jobs=4, random_state=args.seed)
    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.settings import settings
    if version('tabpfn') != '9.1.0': raise RuntimeError('Use pinned TabPFN 9.1.0')
    if not args.checkpoint: raise ValueError('Supply --checkpoint; obtain weights separately from the provider.')
    if digest(args.checkpoint) != config['checkpoint']['sha256']: raise ValueError('Checkpoint hash mismatch')
    # TabPFN identifies format by extension after resolving symlinks.
    if Path(args.checkpoint).resolve().suffix != '.safetensors':
        raise ValueError('Copy the downloaded checkpoint to a real .safetensors file (not an extensionless cache symlink).')
    params = dict(case['tabpfn_params'])
    if args.kind == 'importance':
        provenance = json.loads((ROOT/'results/importance/provenance.json').read_text())
        params = dict(next(p for p in provenance if p['element']==args.element and p['model']==args.model)['params'])
        # The original Cr importance run used the automatic Apple MPS selection.
        params.setdefault('device', 'mps')
    for key, value in params.pop('runtime_options', {}).items():
        if key != 'max_batched_estimator_rows': raise ValueError('Unexpected runtime option')
        setattr(settings.tabpfn, key, value)
    if params['device'] == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Recorded case requires Apple MPS; a different backend is a separate experiment.')
    torch.set_num_threads(4)
    # Start only after torch finishes importing; read atomically in the parent.
    import threading
    def sample_mps():
        while True:
            value = torch.mps.driver_allocated_memory() if params['device']=='mps' else 0
            path = Path(args.output)/'mps_bytes.txt'
            temp = path.with_suffix('.tmp'); temp.write_text(str(value)); temp.replace(path)
            time.sleep(.2)
    threading.Thread(target=sample_mps, daemon=True).start()
    cls = TabPFNClassifier if args.task == 'coord' else TabPFNRegressor
    return cls.create_default_for_version(ModelVersion.V3_5, random_state=args.seed,
               model_path=args.checkpoint, n_preprocessing_jobs=4, **params)


def atomic_npz(path, **arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as file: np.savez_compressed(file, **arrays)
    temp.replace(path)


def importance_worker(args, model, data):
    """Actual inference, not recomputation of importance from bundled predictions.

    Recreate every released donor order. All RF repeats (3) are rerun; the paired
    figure uses the first 1 for non-Cr elements to match their TabPFN evidence.
    Each cell is checkpointed so a time limit need not discard completed work.
    """
    source = ROOT/f'results/importance/{args.element}_{args.model}.npz'
    manifest = json.loads((ROOT/'results/SHA256SUMS.json').read_text())
    if digest(source) != manifest[str(source.relative_to(ROOT))]: raise ValueError('Released source hash mismatch')
    with np.load(source, allow_pickle=False) as z: reference = {k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(data['valid_y'], reference['y'])
    np.testing.assert_array_equal(np.argmax(data['valid_x'], axis=1), reference['white_line_indices'])
    for order in reference['donor_orders']:
        np.testing.assert_array_equal(np.sort(order), np.arange(len(reference['y'])))
    out = Path(args.output)
    fingerprint = {
        'source_sha256':digest(source), 'config_sha256':digest(ROOT/'configs/benchmark.json'),
        'code_sha256':digest(__file__), 'seed':42, 'element':args.element, 'model':args.model,
        'kind':'importance', 'task':'coord', 'packages':package_versions(args.model),
    }
    record = out/'identity.json'
    if record.exists() and json.loads(record.read_text()) != fingerprint:
        raise ValueError('Resume identity changed; preserve previous outputs and use a separate checkout.')
    record.write_text(json.dumps(fingerprint, indent=2)+'\n')
    began = time.monotonic()
    model.fit(data['train_x'], data['train_y'])
    baseline = np.asarray(model.predict(data['valid_x'])).reshape(-1)
    # A mismatch stops rather than silently splicing old/new fitted models.
    np.testing.assert_array_equal(baseline, reference['baseline_prediction'])
    effects = np.empty(reference['permutation_drops'].shape)
    predictions = np.empty(reference['permuted_predictions'].shape, dtype=baseline.dtype)
    for w, (start, stop) in enumerate(reference['windows']):
        for r, order in enumerate(reference['donor_orders']):
            path = out/f'window_{w:02d}_shuffle_{r:02d}.npz'
            check = path.with_suffix('.json')
            if path.exists() and check.exists():
                if digest(path) != json.loads(check.read_text())['sha256']: raise ValueError(f'Checkpoint corrupted: {path.name}')
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['y'], reference['y'])
                    np.testing.assert_array_equal(z['donor_order'], order)
                    pred = z['prediction'].copy()
            else:
                shuffled = data['valid_x'].copy()
                shuffled[:,start:stop] = data['valid_x'][order,start:stop]
                pred = np.asarray(model.predict(shuffled)).reshape(-1)
                del shuffled
                if pred.shape != baseline.shape or not np.isin(pred,[4,5,6]).all(): raise ValueError('Invalid prediction')
                atomic_npz(path, prediction=pred, y=reference['y'], donor_order=order)
                check.write_text(json.dumps({'sha256':digest(path)})+'\n')
            predictions[w,r] = pred
            effects[w,r] = score(reference['y'],baseline,'coord')['macro_f1'] - score(reference['y'],pred,'coord')['macro_f1']
            print(f'Window {w+1}/10, shuffle {r+1}/{len(reference["donor_orders"])} complete', flush=True)
    arrays = {k: v for k,v in reference.items() if k not in ['permutation_drops','permuted_predictions']}
    arrays.update(baseline_prediction=baseline,permutation_drops=effects,permuted_predictions=predictions)
    atomic_npz(out/'importance.npz', **arrays)
    result = dict(fingerprint, status='success',elapsed_seconds=time.monotonic()-began,
                  baseline_exact=True, shuffles=len(reference['donor_orders']),
                  all_shuffled_predictions_exact=bool(np.array_equal(predictions,reference['permuted_predictions'])),
                  maximum_importance_difference=float(np.max(np.abs(effects-reference['permutation_drops']))))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def compare(folder):
    """Compare a fresh rerun with released predictions, including discrepancies."""
    result = json.loads((folder/'result.json').read_text())
    if result.get('status') != 'success': raise ValueError('Run did not complete successfully')
    if result.get('kind') == 'importance':
        return {k:result[k] for k in ['element','model','shuffles','baseline_exact','all_shuffled_predictions_exact','maximum_importance_difference']}
    import csv, gzip
    manifest=json.loads((ROOT/'results/SHA256SUMS.json').read_text())
    if digest(ROOT/'results/test_predictions.csv.gz') != manifest['results/test_predictions.csv.gz']:
        raise ValueError('Released test-prediction checksum mismatch')
    if 'predictions_sha256' in result and digest(folder/'predictions.npz') != result['predictions_sha256']:
        raise ValueError('Rerun prediction checksum mismatch')
    with gzip.open(ROOT/'results/test_predictions.csv.gz','rt') as f:
        rows = [r for r in csv.DictReader(f) if r['element']==result['element'] and r['task']==result['task']
                and r['model']==result['model'] and int(r['seed'])==result['seed']]
    if not rows: return {'released_reference_available':False, 'reason':'No successful released prediction for this case.'}
    rows.sort(key=lambda r:int(r['row']))
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['y'],np.array([float(r['reference']) for r in rows]))
        actual=z['predictions'].copy(); original=np.array([float(r['prediction']) for r in rows])
        measured=score(z['y'],actual,result['task']); expected=score(z['y'],original,result['task'])
    return dict(element=result['element'],task=result['task'],model=result['model'],seed=result['seed'],
                predictions_exact=bool(np.array_equal(actual,original)),
                max_absolute_prediction_difference=float(np.max(np.abs(actual-original))),
                rerun_metrics=measured,released_metrics=expected)
