# CPU port; original experiments/weight_space/white_view_contrasts.py, SHA256 1fcce8d96daf1373de42f2db8ba2e2f080d8b62b8cce29b27269c7aacb9bf2ff
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare a tree's view predictions at the original physical boundary."""
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
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from partition_evidence import OUT, ROOT

def generate(tag, runs):
    with np.load(OUT / 'white_loop_parent.npz') as src:
        parts = [src['ids'], src['tids']]
    groups = {}
    sources = {}
    for name in runs:
        path = ROOT / f'tune_out/ver_{name}.npz'
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as q:
            views = tuple(q['kinds'].astype(str))
            assert views[0] == 'orig' and len(set(views)) == len(views)
            value = np.column_stack([q['oof'], q['test']])
            assert value.shape == (len(views), sum(map(len, parts)))
            groups.setdefault(views, []).append(logit(value.clip(1e-06, 0.999999)))

    def table(part, ids, v, name):
        base = OUT / f'view_{v}' if v else OUT
        if name == 'n':
            path = base / f'clean/features_{part}.parquet' if v else ROOT / f'clean/features_{part}.parquet'
            return pd.read_parquet(path, columns=['n_pre', 'n_post']).loc[ids].to_numpy()
        return pd.read_parquet(base / f'{name}_{part}.parquet').loc[ids]
    each = []
    start = 0
    for part, ids in zip(('train', 'test'), parts):
        sl = slice(start, start + len(ids))
        start += len(ids)
        columns = {}
        n = table(part, ids, '', 'n')
        size = 1 / (1 / n[:, 0] + 1 / n[:, 1])
        wk = table(part, ids, '', 'innov_v2').pre_walk.to_numpy()
        left = table(part, ids, '', 'volatility_law').pre_cluster.to_numpy()
        right = table(part, ids, '', 'volatility_law').post_cluster.to_numpy()
        white = table(part, ids, '', 'innov_v2').pre_white.to_numpy()
        for views, members in groups.items():
            scores = np.mean(members, 0)[:, sl]
            orig = scores[0]
            prefix = '_'.join(views)
            crops = [(i, v) for i, v in enumerate(views) if v.startswith('c')]
            resid = []
            fractions = []
            asym = []
            for i, v in crops:
                dims = table(part, ids, v, 'n')
                ratio = 1 / (1 / dims[:, 0] + 1 / dims[:, 1]) / size
                fractions.append(ratio)
                asym.append(dims[:, 0] / n[:, 0] - dims[:, 1] / n[:, 1])
                resid.append((scores[i] - orig).clip(-5, 5))
            if resid:
                residual = np.stack(resid)
                frac = np.stack(fractions)
                small = residual.mean(0)
                centered = small - orig * (np.sqrt(frac.mean(0)).clip(0.25, 1) - 1)
                for key, z in (('crop', small), ('size_adjusted', centered), ('imbalance', (residual * np.stack(asym)).mean(0)), ('abs', abs(small)), ('disagreement', residual.std(0)), ('direction', np.tanh(orig / 2) * small)):
                    columns[prefix + '_' + key] = z
            if 'rev' in views:
                r = (scores[views.index('rev')] - orig).clip(-6, 6)
                rev_white = table(part, ids, 'rev', 'innov_v2').pre_white.to_numpy()
                for key, z in (('reversal', r), ('length', r * (n[:, 0] - n[:, 1]) / n.sum(1)), ('vol', r * (left - right)), ('white', r * (white - rev_white)), ('walk', r * wk), ('absolute', abs(r)), ('confidence', r * np.tanh(orig / 2))):
                    columns[prefix + '_' + key] = z
        df = pd.DataFrame(columns, index=pd.Index(ids, name='id')).astype(np.float32)
        assert np.isfinite(df).all().all()
        each.append(df)
    assert each[0].columns.equals(each[1].columns)
    for part, a in zip(('train', 'test'), each):
        dest = OUT / f'{tag}_{part}.parquet'
        if dest.exists():
            assert a.equals(pd.read_parquet(dest))
        else:
            a.to_parquet(dest)
    (OUT / (tag + '_sources.json')).write_text(json.dumps(sources, indent=2))
    print(tag, each[0].shape, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tag')
    p.add_argument('runs', nargs='+')
    a = p.parse_args()
    generate(a.tag, a.runs)
