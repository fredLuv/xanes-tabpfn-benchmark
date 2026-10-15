"""Reconstruct headline metrics and regional importance from saved predictions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
ELEMENTS = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def predictions():
    return pd.read_csv(ROOT / 'results/test_predictions.csv.gz', float_precision='round_trip')


def metrics(y, p, task):
    if task == 'coord':
        f = f1_score(y, p, labels=[4, 5, 6], average=None, zero_division=0)
        return dict(accuracy=accuracy_score(y, p), macro_f1=float(f.mean()),
                    **{f'f1_cn{c}': float(v) for c, v in zip([4, 5, 6], f)})
    return dict(mae=mean_absolute_error(y, p), r2=r2_score(y, p))


def metric_table():
    rows = []
    for (element, task, model, seed), d in predictions().groupby(['element', 'task', 'model', 'seed']):
        rows.append(dict(element=element, task=task, model=model, seed=seed,
                         n=len(d), **metrics(d.reference, d.prediction, task)))
    return pd.DataFrame(rows)


def paired_summary(task, metric=None):
    metric = metric or ('macro_f1' if task == 'coord' else 'mae')
    d = metric_table().query('task == @task')
    paired = d.pivot(index=['element', 'seed'], columns='model', values=metric).dropna()
    rows = []
    for element in ELEMENTS:
        v = paired.loc[element]
        rf, tab = v.rf.mean(), v.tabpfn.mean()
        rows.append(dict(Element=element, Seeds=len(v), RF=rf, TabPFN=tab,
                         RF_SD=v.rf.std(ddof=0), TabPFN_SD=v.tabpfn.std(ddof=0),
                         Relative_change_percent=100 * (tab - rf) / rf))
    return pd.DataFrame(rows).set_index('Element')


def importance(element, model):
    with np.load(ROOT / f'results/importance/{element}_{model}.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    repeats = 3 if element == 'Cr' else 1
    base = metrics(d['y'], d['baseline_prediction'], 'coord')['macro_f1']
    effects = np.array([[base - metrics(d['y'], pred, 'coord')['macro_f1']
                         for pred in window[:repeats]] for window in d['permuted_predictions']])
    np.testing.assert_allclose(effects, d['permutation_drops'][:, :repeats], atol=1e-12, rtol=0)
    return d, effects


def importance_table():
    rows = []
    for element in ELEMENTS:
        for model in ['rf', 'tabpfn']:
            d, effects = importance(element, model)
            j = int(effects.mean(1).argmax()); a, b = d['windows'][j]
            mode = np.bincount(d['white_line_indices'], minlength=100).argmax()
            rows.append(dict(Element=element, Model=model, Shuffles=effects.shape[1],
                             Top_region_eV=f"{d['energy_ev'][a]:.1f}–{d['energy_ev'][b-1]:.1f}",
                             Macro_F1_decrease=effects.mean(1)[j], Modal_maximum_eV=d['energy_ev'][mode]))
    return pd.DataFrame(rows)


def verify():
    manifest = json.loads((ROOT / 'results/SHA256SUMS.json').read_text())
    for name, digest in manifest.items():
        assert sha256(ROOT / name) == digest, f'Checksum mismatch: {name}'
    frame = predictions()
    assert not frame.duplicated(['element', 'task', 'model', 'seed', 'row']).any()
    table = metric_table()
    assert len(table) == 143
    assert table.groupby('model').size().to_dict() == {'rf': 72, 'tabpfn': 71}
    records = json.loads((ROOT / 'results/run_records.json').read_text())
    for r in records:
        if r['status'] != 'success': continue
        v = table[
            (table.element == r['element']) & (table.task == r['task']) &
            (table.model == r['model']) & (table.seed == r['seed'])].iloc[0]
        expected = r['metrics'].copy()
        if 'per_class_f1' in expected:
            for c, x in zip([4, 5, 6], expected.pop('per_class_f1')): expected[f'f1_cn{c}'] = x
        for k, x in expected.items():
            if k in table.columns: np.testing.assert_allclose(v[k], x, atol=1e-12, rtol=0)
    for (_, _, _), d in frame.groupby(['element', 'task', 'seed']):
        if d.model.nunique() == 2:
            rf = d[d.model == 'rf'].sort_values('row')
            tab = d[d.model == 'tabpfn'].sort_values('row')
            np.testing.assert_array_equal(rf.row, tab.row)
            np.testing.assert_array_equal(rf.reference, tab.reference)
    profiles = json.loads((ROOT / 'results/coordination_profiles.json').read_text())
    for element in ELEMENTS:
        panel = next(v for v in profiles['panels'] if v['element'] == element)
        rf, _ = importance(element, 'rf'); tab, _ = importance(element, 'tabpfn')
        for key in ['y', 'windows', 'energy_ev', 'white_line_indices']:
            np.testing.assert_array_equal(rf[key], tab[key])
        repeats = 3 if element == 'Cr' else 1
        np.testing.assert_array_equal(rf['donor_orders'][:repeats], tab['donor_orders'][:repeats])
        for model in ['rf', 'tabpfn']:
            d, effects = importance(element, model)
            for order in d['donor_orders']: np.testing.assert_array_equal(np.sort(order), np.arange(len(d['y'])))
            np.testing.assert_allclose(effects.mean(1), panel['models'][model]['importance'], atol=1e-12, rtol=0)
    return 'Verified 143 test runs, matched test rows, and all 16 coordination importance profiles.'
