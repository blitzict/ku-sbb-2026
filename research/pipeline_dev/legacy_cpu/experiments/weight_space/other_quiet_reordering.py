# CPU port; original experiments/weight_space/other_quiet_reordering.py, SHA256 0547421f30952266a37e0c5b2bb03ca47b0dc30dab2f530a0b143c19da0d07ce
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Reorder only low-scoring series for which no recent detector raises a flag."""
import contextlib as _cpu_context
import os as _cpu_os
from pathlib import Path as _CpuPath

def _cpu_workspace():
    value = _cpu_os.environ.get('KU_SBB_ROOT')
    if not value:
        raise RuntimeError('Set KU_SBB_ROOT to the fresh CPU rebuild workspace')
    root = _CpuPath(value).resolve()
    if not (root / 'CPU_REBUILD_WORKSPACE.json').is_file():
        raise RuntimeError('CPU ports require a marked rebuild workspace')
    return root
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata
from other_miss_witnesses import make
from other_archived_votes import load
from other_response_gate import run
from partition_evidence import OUT, ROOT
from white_local_rank_weights import centered, weights
from threadpoolctl import threadpool_limits

def setup(ids, tids, y, serial_only=False, legacy_vote=False):
    panels, paths = make(ids, tids)
    panel = np.column_stack([s.mean(1) for s in panels])

    def measurement(table, pattern):
        values = []
        for part, ix in (('train', ids), ('test', tids)):
            p = OUT / f'{table}_{part}.parquet'
            paths.append(p)
            t = pd.read_parquet(p).loc[ix].filter(regex=pattern)
            assert len(t.columns) > 0
            values.append(t.to_numpy())
        return np.vstack(values)
    neighbors = measurement('transition_neighbors', '_held_k12_separation$').mean(1)
    memory = measurement('memory_law1', '^unit_start$')[:, 0]
    timing = (measurement('kernel_timing', '_center_advantage$') / (1 + measurement('kernel_timing', '_residual_shape$'))).mean(1)
    heads = []
    for s in (7, 29):
        p = OUT / f'process_identity_s{s}.npz'
        paths.append(p)
        with np.load(p) as a:
            assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
            heads.append(np.vstack([a['oof'], a['test']]))
    between, _, pre, post = np.hsplit(np.mean(heads, axis=0)[:, 2:], 4)
    identity = (between - 0.5 * (pre + post)).mean(1)
    signals = np.column_stack([neighbors, identity, timing, memory])
    ix = ids.append(tids)
    gate = np.ones(len(ix), bool)
    if legacy_vote:
        older, old_files = load(ids, tids)
        paths.extend(old_files)
    if serial_only:
        path = ROOT / 'tune_out/generator_segments.parquet'
        paths.append(path)
        gen = pd.read_parquet(path, columns=['id', 'part', 'p_bic', 'phi1_ar1'])
        pre = gen.loc[gen.part == 'pre'].set_index('id').loc[ix]
        gate = ((pre.p_bic != 0) | (pre.phi1_ar1 > 0.95)).to_numpy()
    checks = {}
    for grp in ('other_disagreement', 'other_no_detector'):
        pair = pd.read_csv(OUT / f'{grp}_pairs.csv', dtype={'case_id': str, 'control_id': str})
        pos = ix.get_indexer(pair.case_id)
        neg = ix.get_indexer(pair.control_id)
        rank = rankdata(signals[np.r_[pos, neg]], axis=0)
        gap = rank[:len(pos)] - rank[len(pos):]
        checks[grp] = dict(n=len(pos), wins=int(sum(gap.mean(1) > 0)), ties=int(sum(gap.mean(1) == 0)), gap_correlations=np.corrcoef(gap.T).round(3).tolist())
    print('prespecified mean rank', checks, flush=True)
    (OUT / 'other_quiet_signals.json').write_text(json.dumps(checks, indent=2))

    def fit(tr, va, z, v, prior, prev):
        ids_both = np.r_[tr, va]
        base = np.r_[z, v]
        n = len(tr)
        sel = y[tr] == 0
        eligible = gate[ids_both] & (base < np.quantile(z[sel], 0.8)) & np.all(panel[ids_both] < np.quantile(panel[tr[sel]], 0.8, axis=0), axis=1)
        if legacy_vote:
            eligible &= np.any(older[ids_both] >= np.quantile(older[tr[sel]], 0.8, axis=0), axis=1)
        refs = np.flatnonzero(sel & eligible[:n])
        r = np.column_stack([np.searchsorted(np.sort(signals[tr[refs], k]), signals[ids_both, k]) / len(refs) for k in range(signals.shape[1])]).mean(1)
        _, center = centered(r[refs, None], base[refs])
        d = centered(r[:, None], base, center)[0] * eligible[:, None] * 3
        w = float(np.clip(weights(d[:n], z, y[tr])[0], 0, 0.55))
        provisional = base + w * d[:, 0]
        grid = np.r_[0.0005, 0.002, 0.007, np.linspace(0.02, 0.98, 49), 0.993, 0.998, 0.9995]
        orig = np.quantile(base[refs], grid)
        change = np.quantile(provisional[refs], grid)
        bias = gaussian_filter1d(orig - change, 0.95)
        fit.parameters = dict(signal_weight=w, serial_only=serial_only, legacy_vote=legacy_vote, negative_reference_sources=int(len(refs)), risk_cutoff=float(np.quantile(z[sel], 0.8)), detector_cutoffs=np.quantile(panel[tr[sel]], 0.8, axis=0).tolist(), provisional_quantiles=change.tolist(), negative_distribution_offset=bias.tolist())
        val = provisional[n:] + eligible[n:] * np.interp(provisional[n:], change, bias)
        val = np.where(eligible[n:], val, v)
        return (val, (val - v)[:, None], np.ones(1))
    return (fit, [*paths, Path(__file__).resolve()])
if __name__ == '__main__':
    serial = 'serial' in sys.argv[1:]
    legacy = 'legacy' in sys.argv[1:]
    with threadpool_limits(limits=1):
        run(lambda i, t, y: setup(i, t, y, serial, legacy), 'local_rank_quiet_reordering' + ('_serial' if serial else '') + ('_legacy' if legacy else ''))
