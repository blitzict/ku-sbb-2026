# CPU port; original experiments/weight_space/white_memory_replication.py, SHA256 10b97a79fde04453a1af13d941c0e58c2b46b98f73ed9f6119ab89d669e27175
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test the fixed onset-or-cessation contrast outside the cases that suggested it."""
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
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from white_signature_screen import pairs
from partition_evidence import ROOT, OUT
import white_loop as wl

def main():
    ids, positive, control, matching = pairs()
    _, current, _, y, *_ = wl.load()
    assert current.equals(ids)
    p, q = [np.load(ROOT / f'tune_out/oof_{t}.npy') for t in ('MECH3C12_s0', 'MECH3C12V_s0')]
    selected = np.flatnonzero((y == 1) & (p < np.quantile(p[y == 0], 0.8)) & (q >= np.quantile(q[y == 0], 0.8)))
    neg = np.flatnonzero(y == 0)
    used = [selected]
    for prob in (p, q):
        z = logit(prob.clip(1e-06, 1 - 1e-06))
        a, b = linear_sum_assignment(abs(z[selected, None] - z[neg][None, :]))
        assert len(a) == 162
        used.append(neg[b])
    exclude = np.unique(np.concatenate(used))
    keep = ~(np.isin(positive, exclude) | np.isin(control, exclude))
    positive, control = (positive[keep], control[keep])
    frame = pd.read_parquet(OUT / 'memory_events_train.parquet').loc[ids]
    switch = (frame.prob_on + frame.prob_off).to_numpy()
    d = switch[positive] - switch[control]
    rng = np.random.default_rng(59147)
    perm = np.concatenate([(rng.integers(2, size=(512, len(d)), dtype=np.int8) * 2 - 1) @ d / len(d) for _ in range(64)])
    boots = np.concatenate([d[rng.integers(len(d), size=(256, len(d)))].mean(1) for _ in range(64)])
    out = dict(n=len(d), distinct_controls=len(np.unique(control)), previously_matched=len(keep), excluded_pos=int(np.isin(selected, np.r_[positive, control]).sum()), excluded_pairs=int(sum(~keep)), case=float(switch[positive].mean()), control=float(switch[control].mean()), mean_difference=float(d.mean()), pairwise_win_fraction=float(np.mean((d > 0) + 0.5 * (d == 0))), median_difference=float(np.median(d)), all_case_control_auc=float(roc_auc_score(np.r_[np.ones(len(d)), np.zeros(len(d))], switch[np.r_[positive, control]])), paired_interval=np.quantile(boots, [0.025, 0.975]).tolist(), one_sided_p=float((1 + sum(perm >= d.mean())) / (1 + len(perm))), scope='Fixed combined probability of clustering onset or cessation. Removed all recovered positives and both sets of their distinct score-matched controls before evaluation. Same underlying competition population and physical program.', original_matching=matching)
    assert out['excluded_pos'] == 0
    (OUT / 'memory_switch_replication.json').write_text(json.dumps(out, indent=2))
    pd.DataFrame(dict(positive_id=ids[positive], control_id=ids[control], positive_switch=switch[positive], control_switch=switch[control])).to_csv(OUT / 'memory_switch_replication_pairs.csv', index=False)
    print(json.dumps(out, indent=1), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
