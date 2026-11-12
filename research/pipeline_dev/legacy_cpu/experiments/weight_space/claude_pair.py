# CPU port; original experiments/weight_space/claude_pair.py, SHA256 9d092826e510df752c772b0b0a4a98b95216876d71ed1e1c21dcf8307e3a13e1
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare a collaborator's matched base and new OOF runs against the live anchor."""
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
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits
import expert_fusion as ef
from residual_trial import baseline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base', type=Path)
    ap.add_argument('new', type=Path)
    args = ap.parse_args()
    data = ef.read_data()
    _, ids, _, y, wt, *_ = data

    def load(p):
        x = np.load(p, allow_pickle=False)
        assert x.shape == (len(ids),) and np.isfinite(x).all() and np.all((x >= 0) & (x <= 1))
        return logit(np.clip(x.astype(float), 1e-06, 1 - 1e-06))
    base, new = (load(args.base), load(args.new))
    folds, _ = baseline(data)
    results = {}
    for split, parts in folds.items():
        pred = {k: np.zeros(len(y)) for k in ['anchor', 'base', 'window', 'paired_delta']}
        for tr, va, zr, zv in parts:
            pred['anchor'][va] = zv
            for mode in ['base', 'window', 'paired_delta']:
                f = {'base': base, 'window': new, 'paired_delta': new - base}[mode]
                fr, fv = (f[tr], f[va])
                if mode != 'paired_delta':
                    fr, fv = (fr - zr, fv - zv)
                X = np.column_stack([fr, np.clip(fr, -3.0, 3.0) * np.tanh(zr / 2)])
                T = np.column_stack([fv, np.clip(fv, -3.0, 3.0) * np.tanh(zv / 2)])
                th = ef.fit_head(X, zr, y[tr])
                pred[mode][va] = zv + T @ th[:-1] + th[-1]
        results[split] = {k: ef.score(y, p, wt) for k, p in pred.items()}
    report = {'base': str(args.base), 'base_sha': ef.sha(args.base), 'new': str(args.new), 'new_sha': ef.sha(args.new), 'single': ef.score(y, new, wt), 'results': results}
    (ef.OUT / f'paired_{args.new.stem}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
