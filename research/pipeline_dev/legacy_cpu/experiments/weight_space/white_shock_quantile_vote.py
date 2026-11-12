# CPU port; original experiments/weight_space/white_shock_quantile_vote.py, SHA256 795ce4e7ec85a1947d28254fd79d5059fdb00ddf337fa5ad5b6b6044891a1973
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure the quantile-response head against the simultaneously fitted physical-only head."""
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
import argparse
import numpy as np
import pandas as pd
from white_joint import read_member
from white_nested import audit
from partition_evidence import OUT
import white_loop as wl

def main(kind):
    _, ids, tids, y, wt, wn, miss = wl.load()
    control, changed = ('law_path_control', 'law_ordered_paths') if kind == 'ordered_path' else ('law_shock_quantile_control', 'law_shock_quantiles')
    baseline = read_member(control, ids, tids)[0][:, 0]
    curve = read_member(changed, ids, tids)[0][:, 0]
    n = len(ids)
    result = {k: wl.score(y, p[:n], wt, wn, miss) for k, p in (('physical', baseline), ('with_curves', curve))}
    result['recovery'] = audit(y, miss, baseline[:n], curve[:n])
    delta = curve - baseline
    frame = pd.DataFrame(dict(response=delta), index=ids.append(tids))
    for tag, col in (('volatility_law', 'pre_cluster'), ('mode_residual', 'gate')):
        gate = np.concatenate([pd.read_parquet(OUT / f'{tag}_{p}.parquet', columns=[col]).loc[ix, col] for p, ix in (('train', ids), ('test', tids))])
        frame[tag] = delta * gate
    for p, ix in (('train', ids), ('test', tids)):
        frame.loc[ix].to_parquet(OUT / f'{kind}_vote_{p}.parquet')
    (OUT / f'{kind}_head_pair.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', nargs='?', default='shock_quantile', choices=('shock_quantile', 'ordered_path'))
    main(p.parse_args().kind)
