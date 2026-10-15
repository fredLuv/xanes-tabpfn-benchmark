"""Recheck saved-model row-resampling sensitivity; no new donor shuffles."""
import json
import numpy as np
from scipy.stats import rankdata
from .analysis import ROOT, ELEMENTS, importance

def scores(y, predictions, weights):
    """Macro-F1 for each resample and prediction vector from weighted counts."""
    y = np.asarray(y, dtype=int) - 4
    p = np.asarray(predictions, dtype=int) - 4
    assert np.isin(y, [0, 1, 2]).all() and np.isin(p, [0, 1, 2]).all()
    categories = y[None, :] * 3 + p
    indicators = (categories[:, :, None] == np.arange(9)).transpose(1, 0, 2)
    counts = (weights @ indicators.reshape(len(y), -1).astype(float)).reshape(len(weights), len(p), 3, 3)
    tp = np.diagonal(counts, axis1=2, axis2=3)
    denominator = counts.sum(axis=2) + counts.sum(axis=3)
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0).mean(axis=2)

def correlation(a, b):
    a, b = rankdata(a), rankdata(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])

def verify_stability():
    saved=json.loads((ROOT/'results/conditional_stability.json').read_text())
    rng=np.random.default_rng(20261014)
    for element in ELEMENTS:
        d,_=importance(element,'rf'); y=d['y']; n=len(y)
        weights=np.zeros((1000,n))
        for label in [4,5,6]:
            idx=np.flatnonzero(y==label)
            weights[:,idx]=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=1000)
        repeats=3 if element=='Cr' else 1
        for model in ['rf','tabpfn']:
            d,drops=importance(element,model)
            p=np.concatenate([d['baseline_prediction'][None],d['permuted_predictions'][:,:repeats].reshape(-1,n)])
            s=scores(y,p,weights)
            effects=(s[:,:1]-s[:,1:]).reshape(1000,10,repeats).mean(2)
            tops=effects==effects.max(1,keepdims=True)
            observed=drops.mean(1); original=np.flatnonzero(observed==observed.max())
            record=next(r for r in saved['rows'] if r['element']==element and r['model']==model)
            np.testing.assert_allclose(tops[:,original].any(1).mean(),record['observed_top_retained_fraction'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.quantile(effects,[.025,.975],axis=0).T,record['conditional_effect_percentiles'],atol=1e-12,rtol=0)
    return 'Reconstructed all 16 conditional resampling summaries (1,000 resamples each).'

if __name__=='__main__': print(verify_stability())
