# CPU port; original experiments/weight_space/white_baseline_opinions.py, SHA256 329b5957601d0c648c125b305f98e833e550f54f7b170a2c2df694a1068d1819
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare the initial frozen detector families with the current physical detectors."""
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
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import white_local_rank_weights as rank
from white_expert_agreement import GROUPS, constructor
from white_joint import read_member
FAMILIES = (('v6_lr3_xlg', 'v6_lr3_xlg_s1', 'v6_lr3_xlg_s2', 'v6_lr3_xlg_s3', 'v6_lr3_xlg_s4'), ('v6_lr5_xlg', 'v6_lr5_xlg_s1', 'v6_bf_xlg', 'v6_gh_xlg'), ('L3_rank_lr03_s0', 'L3_rank_lr03_s1'), ('L4_fam_grid_b2', 'L4_fam_dd_b2', 'L4_fam_lr_b2', 'L4_fam_gh_b2'), ('VARC4hv_ns3_wc', 'VARC4_ns3_wc', 'VARC4_lm_wc', 'VARC4_full_wc2'), ('MLPHV_bag_w', 'W_SEQ_qb5'))

def inputs(ids, tids, models, physics):
    make, paths = constructor([])(ids, tids, models, physics)
    data = ef.read_data()
    assert ids.equals(data[1]) and tids.equals(data[2])
    names = json.loads((ef.ROOT / 'submissions/sub_seqw5_stack.json').read_text())['members']
    path = lambda s, n: ef.OLD / f'{s}_head_v2_bag.npy' if n == 'MLPHV_bag_w' else ef.O / f'{s}_{n}.npy'
    old = []
    scatter = []
    for family in FAMILIES:
        panels = []
        for n in family:
            tr = data[5][:, names.index(n)]
            pp = path('test', n)
            paths.extend([path('oof', n), pp])
            panels.append(np.r_[tr, logit(np.load(pp).clip(1e-06, 1 - 1e-06))])
        u = np.column_stack(panels)
        old.append(u.mean(1))
        scatter.append(u.std(1))
    old = np.column_stack(old)
    sc = np.column_stack(scatter)
    new = np.column_stack([read_member(s, ids, tids)[0][:, 0] for s in ('claude_cropvolbag', 'lawforecast', 'lawrank_crops')])
    ctx = np.nan_to_num(data[11][:, :data[12]], nan=0, posinf=20, neginf=-20)

    def feats(ix, z):
        b = make(ix, z)
        h = np.tanh((old[ix] - z[:, None]) / 2)
        d = np.tanh((old[ix] - np.mean(new[ix], 1, keepdims=True)) / 1.6)
        increment = np.mean(new[ix], 1) - old[ix].mean(1)
        f = [b, 0.8 * h, 0.8 * d, 0.62 * sc[ix], 0.6 * h * abs(d), 0.45 * np.tanh(increment[:, None]) * sc[ix], 0.65 * (h - h.mean(1, keepdims=True)) * (1 - np.tanh(z[:, None] / 2) ** 2)]
        for c, width in ((-2.0, 0.88), (-0.5, 0.78), (0.75, 1.05)):
            window = np.exp(-0.5 * ((z[:, None] - c) / width) ** 2)
            f.extend([0.75 * window * h, 0.55 * window * d])
        g = np.column_stack([np.tanh(ctx[ix, 2]), np.tanh(ctx[ix, 3] / 3), np.tanh(ctx[ix, 5] / 2)])
        f.append(0.52 * (g[:, :, None] * d[:, None, :]).reshape(len(ix), -1))
        return np.ascontiguousarray(np.column_stack(f))
    paths.extend([Path(__file__).resolve(), Path(ef.__file__).resolve(), ef.BASE, ef.ROOT / 'submissions/sub_seqw5_stack.json'])
    paths.extend((ef.ROOT / f'{stem}_{part}.parquet' for stem in ('clean/features', 'tune_out/lr5') for part in ('train', 'test')))
    return (feats, list(dict.fromkeys(paths)))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tag')
    p.add_argument('extra', nargs='*')
    a = p.parse_args()
    models = [*GROUPS['crop_trees'][:3], *GROUPS['rank_trees'], 'lawforecast', 'lawretrieval', *a.extra]
    with threadpool_limits(limits=1), patch.object(rank, 'inputs', inputs):
        rank.run('reversible_rank', a.tag, models, True)
