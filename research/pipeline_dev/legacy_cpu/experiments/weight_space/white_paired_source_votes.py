# CPU port; original experiments/weight_space/white_paired_source_votes.py, SHA256 3cffd4346322cda7374a4dcdec75707d083f2d069fd2842eba5682bcfa4db290
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Isolate what a physical-feature mask and a second rank-pattern fit change."""
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
import numpy as np
import pandas as pd
from white_joint import read_member
from partition_evidence import OUT
import white_loop as wl

def main():
    _, ids, tids, y, wt, wn, miss = wl.load()
    n = len(ids)

    def member(name):
        return read_member(name, ids, tids)[0][:, 0]

    def table(tag, col):
        return np.concatenate([pd.read_parquet(OUT / f'{tag}_{p}.parquet', columns=[col]).loc[ix, col] for p, ix in (('train', ids), ('test', tids))])
    mask = member('claude_masked_physics') - member('claude_source_loss')
    ranks = member('law_rank_dependence_s7') - member('law_rank_dependence_s29')
    phase = table('periodic', 'gate')
    cluster = table('volatility_law', 'pre_cluster')
    df = pd.DataFrame(dict(mask=mask, mask_phase=mask * phase, mask_cluster=mask * cluster, ranks=ranks, ranks_phase=ranks * phase, ranks_cluster=ranks * cluster), index=ids.append(tids))
    assert df.index.is_unique and np.isfinite(df).all().all()
    for part, ix in (('train', ids), ('test', tids)):
        df.loc[ix].to_parquet(OUT / f'paired_source_votes_{part}.parquet')
    with np.load(OUT / 'local_rank_opinion_linear_oof.npz') as a:
        assert np.array_equal(a['ids'], ids)
        for seed in (7, 29, 0, 123, 2024, 11041):
            v = a[f's{seed}']
            ref = wl.score(y, v, wt, wn, miss)
            test = wl.score(y, v + 0.2 * mask[:n], wt, wn, miss)
            print(seed, {k: round(test[k] - ref[k], 7) for k in ('auc', 'remaining', 'target_vs_neg')}, flush=True)
if __name__ == '__main__':
    main()
