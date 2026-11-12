# CPU port; original experiments/weight_space/white_named_signals.py, SHA256 18583af0778d91e6ef177d251e652c0f8840f0d64dbe514773161c478bdb7ada
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Validate named sources of a grouped tree vote and isolate their contributions."""
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
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
import white_loop as wl
FEATURES = {'clock': ('phase_law', ['mixture_evidence', 'phase_bf', 'max_evidence', 'common_phase']), 'decay': ('volatility_law', ['pre_decay1', 'pre_decay2', 'pooled_cluster']), 'cluster_scale': ('volatility_law', [f'scale_cluster{i}' for i in range(5)]), 'noise_density': ('volatility_law', ['law3_null', 'scale_and_law', 'all'])}

def run():
    _, ids, tids, y, wt, wn, miss = wl.load()
    ref = np.load(OUT / 'local_rank_three_source_votes_oof.npz')
    rows = []
    for p, ix in (('train', ids), ('test', tids)):
        grouped = pd.read_parquet(OUT / f'tree_channels_claude_{p}.parquet').loc[ix]
        seq = []
        for s in (29, 7):
            src = ROOT / f'tune_out/redesign/featshap_MECH3C12V_srank_k{s}_s0{('' if p == 'train' else '_test')}.parquet'
            x = pd.read_parquet(src).loc[ix]
            assert x.index.is_unique and np.isfinite(x).all().all()
            seq.append(x)
        a = (seq[0] + seq[1]) * 0.5
        g = pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'].to_numpy()
        phase = pd.read_parquet(OUT / f'phase_law_{p}.parquet').loc[ix]
        active = phase.df.to_numpy() > 0
        data = {}
        for name, (source, cols) in FEATURES.items():
            all_shap = a.filter(like=source + '__')
            assert np.max(abs(all_shap.sum(axis=1) - grouped['full_' + source])) < 3e-05
            features = a[[source + '__' + c for c in cols]].to_numpy()
            data[name + '_vote'] = features.sum(1)
            if name == 'clock':
                data[name + '_active'] = active.astype(float)
                data[name + '_direct'] = active * a['phase_law__phase_bf'].to_numpy()
            for col, v in zip(cols, features.T):
                data[source + '__' + col] = v
        data['clock_given_wave'] = data['clock_vote'] * active
        data['volatility_without_wave'] = (data['decay_vote'] + data['cluster_scale_vote']) * ~active
        data['joint'] = np.minimum(data['decay_vote'], data['cluster_scale_vote'])
        tab = pd.DataFrame(data, index=ix)
        tab.to_parquet(OUT / f'named_signals_{p}.parquet')
        rest = pd.read_parquet(OUT / f'tree_channels_shared_{p}.parquet').loc[ix]
        pd.concat([rest, tab.add_prefix('full_named_')], axis=1).to_parquet(OUT / f'tree_channels_named_{p}.parquet')
        if p == 'train':
            n = pd.read_parquet(ROOT / 'clean/features_train.parquet', columns=['n_pre', 'n_post']).loc[ix].sum(axis=1).to_numpy()
            white = np.digitize(g, [0.65, 0.85, 0.94])
            cell = 3 * (4 * np.digitize(ref['s7'], np.quantile(ref['s7'], np.linspace(0, 1, 13)[1:-1])) + white) + np.digitize(n, np.quantile(n, [0.33, 0.67]))
            for tag, mask in (('all', np.ones(len(ix), bool)), ('phase_fit_inactive', ~active), ('phase_fit_active', active)):
                ng = (y == 0) & mask
                mg = miss & mask
                nc = np.bincount(cell[ng], minlength=144)
                pc = np.bincount(cell[mg], minlength=144)
                ok = (nc >= 3) & (pc > 0)
                h = mg & ok[cell]
                c = ng & ok[cell]
                if not np.any(h):
                    continue
                w = pc[cell[c]] / nc[cell[c]]
                for name, v in data.items():
                    rows.append(dict(population=tag, feature=name, historical=int(h.sum()), controls=int(c.sum()), difference=float(v[h].mean() - np.average(v[c], weights=w))))
            votes = phase[['level_bf', 'amplitude_bf', 'phase_bf', 'scale_bf', 'shape_bf']].to_numpy() + np.log([0.22, 0.12, 0.18, 0.27, 0.11])
            which = np.column_stack([votes, np.full(len(ix), np.log(0.1))]).argmax(1)
            fitted = miss & active
            strong = fitted & (phase.mixture_evidence.to_numpy() > 5) & (which == 2)
            print('old phase observations', fitted.sum(), 'strong phase changes', strong.sum(), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'named_signals_matched.csv', index=False)
    print(df[(df.population == 'phase_fit_inactive') & df.feature.isin(['clock_vote', 'decay_vote', 'cluster_scale_vote', 'noise_density_vote'])].to_string(index=False), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
