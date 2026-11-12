# CPU port; original experiments/weight_space/white_paired_audit.py, SHA256 99a482438ea80f0b1ca4ea2329a4a2e9986f323542a82dfcc8053ced5eeb875d
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Recover ties and local paired AUC uncertainty for a fixed set of OOF scores."""
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
import numpy as np
from white_loop import load
from white_nested import audit
from partition_evidence import OUT, ROOT

def contributions(y, s):
    a, b = (s[y == 1], s[y == 0])
    aa, bb = (np.sort(a), np.sort(b))
    positive = 0.5 * (np.searchsorted(bb, a, 'left') + np.searchsorted(bb, a, 'right')) / len(bb)
    negative = 1 - 0.5 * (np.searchsorted(aa, b, 'left') + np.searchsorted(aa, b, 'right')) / len(aa)
    assert abs(positive.mean() - negative.mean()) < 1e-12
    return (positive, negative)

def report(tag):
    q, ids, _, y, wt, wn, miss = load()
    archive = tag if tag.startswith('local_rank_') else 'wn_stage_' + tag
    name = 'wn_' + archive if tag.startswith('local_rank_') else archive
    with np.load(OUT / (archive + '_oof.npz')) as a:
        assert np.array_equal(ids, a['ids'])
        candidate = a['s7']
    parent = np.empty(len(y))
    for f in range(5):
        k = f's7_f{f}_'
        parent[q[k + 'va']] = q[k + 'zv']
    metric = {}
    for cohort, mask in (('all', np.ones(len(y), bool)), ('remaining', ~miss), ('historical_vs_all_negatives', miss | (y == 0))):
        reference = contributions(y[mask], parent[mask])
        now = contributions(y[mask], candidate[mask])
        a, b = [x - z for x, z in zip(now, reference)]
        d = a.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        metric[cohort] = dict(parent=reference[0].mean(), candidate=now[0].mean(), delta=d, fixed_score_interval_95=[d - 1.96 * se, d + 1.96 * se])
    share = miss.sum() / sum(y == 1)
    rest = metric['remaining']['candidate']
    hard = metric['historical_vs_all_negatives']['candidate']
    assert abs(share * hard + (1 - share) * rest - metric['all']['candidate']) < 1e-12
    out = dict(primary=metric, fixed_false_positive_rates=audit(y, miss, parent, candidate), hard_auc_needed_if_remaining_holds=(0.88 - (1 - share) * rest) / share, pairs_needed=int(np.ceil((0.88 - metric['all']['candidate']) * sum(y == 0) * sum(y == 1) - 1e-05)), scope='Paired fixed-score uncertainty on the reused local cohort. Excludes training, model selection and hidden-test uncertainty.')
    dest = ROOT / f'new weights/{name}_candidate/paired_audit.json'
    assert dest.parent.is_dir()
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tag')
    report(p.parse_args().tag)
