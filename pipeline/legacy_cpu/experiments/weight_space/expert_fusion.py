# Fixed winning-recipe CPU adaptation v1
"""Fold-local residual heads for the current packaged ensemble. Uses two CPU threads."""
import rebuild_policy as _rebuild
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
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
ROOT = _cpu_workspace()
sys.path.insert(0, str(ROOT / 'experiments/weight_edit_20261104'))
from view_weight_edit import expand, fit_views
from stack_edit import logistic_fit, score, sha
from finalize import write_submission
O = ROOT / 'tune_out'
OLD = O / 'weight_edit_20261104'
OUT = O / 'weight_space'
PACKAGE = ROOT / 'new weights'
BASE = PACKAGE / 'ensemble.json'
EXPERTS = ['L3_rank_lr03_s0', 'L4_fam_gh_b2', 'VARC4hv_ns3_wc', 'VARC4_lm_wc', 'VARC4_full_wc2', 'MLPHV_bag_w', 'W_SEQ_qb5']
CONTEXT = {'clean/features': ['n_pre', 'n_post', 'acf1_pre', 'kurt_pre', 'skew_pre'], 'tune_out/lr5': ['lr_p_bic', 'lr_persist', 'lr_df_pre', 'lr_bim_sep_pre', 'lr_skew_q_pre'], 'tune_out/gh': ['gh_garch_vs_const_pre']}
EVIDENCE = {'tune_out/lr5': ['lr_level_lbf_t', 'lr_scale_lbf_t', 'lr_ar_lr', 'lr_gn_shape_lr_sym', 'lr_skew_shape_lr_sym', 'lr_bim_shape_lr_sym', 'lr_level_lbf_skew', 'lr_scale_lbf_skew'], 'tune_out/gh': ['gh_e_cvm', 'gh_var_change_lr', 'gh_ab_change_lr', 'gh_agn_shape_lr_sym'], 'tune_out/lm': ['lm_z_ar1', 'lm_z_ar2', 'lm_z_ar3', 'lm_z_lsc', 'lm_z_rlev'], 'tune_out/ps': ['ps_gmm_lr', 'ps_adf_d', 'ps_kim_lr', 'ps_dc1_d', 'ps_l3_d', 'ps_l4_d']}
MODES = _rebuild.select(('regime',), ('expert', 'regime', 'evidence', 'joint'))
SPLITS = (7, 29, 0)

def read_data():
    item = json.loads(BASE.read_text())
    for name, h in item['input_hashes'].items():
        assert sha(ROOT / name) == h, name
    ids, tids = [pd.read_parquet(ROOT / f'clean/features_{p}.parquet', columns=[]).index for p in ('train', 'test')]
    names = json.loads((ROOT / 'submissions/sub_seqw5_stack.json').read_text())['members']
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    wt = _rebuild.diagnostics(O, len(y))
    mats = []
    for part in ('oof', 'test'):
        paths = [OLD / f'{part}_head_v2_bag.npy' if n == 'MLPHV_bag_w' else O / f'{part}_{n}.npy' for n in names]
        mats.append(np.column_stack([logit(np.clip(np.load(p), 1e-06, 1 - 1e-06)) for p in paths]).astype(float))
    archives = {t: OLD / f'ver_head_v2_{t}.npz' for t in ['MLPHV_big', 'MLPHV_big_o1', 'MLPHV_big_o2']}
    V, VT, labels, groups, _ = expand(names, *mats, archives)
    experts = np.vstack(mats)[:, [names.index(n) for n in EXPERTS]]

    def descriptors(spec):
        sides = []
        for part, index in (('train', ids), ('test', tids)):
            blocks = [pd.read_parquet(ROOT / f'{stem}_{part}.parquet', columns=cols).loc[index, cols] for stem, cols in spec.items()]
            assert all((b.index.equals(index) and b.index.is_unique for b in blocks))
            sides.append(np.column_stack(blocks))
        return np.vstack(sides)
    context = descriptors(CONTEXT)
    evidence = descriptors(EVIDENCE)
    assert np.isfinite(experts).all()
    return (item, ids, tids, y, wt, mats[0], V, VT, labels, groups, experts, np.column_stack([context, evidence]), context.shape[1])

def distribution_maps(D):
    return np.nanquantile(np.where(np.isfinite(D), D, np.nan), np.linspace(0, 1, 65), axis=0)

def ranks(D, refs):
    result = []
    for x, q in zip(D.T, refs.T):
        uq, first, count = np.unique(q, return_index=True, return_counts=True)
        grid = (first + 0.5 * (count - 1)) / (len(q) - 1) * 2 - 1
        r = np.interp(x, uq, grid, left=-1.0, right=1.0)
        result.append(np.where(np.isfinite(x), r, 0))
    return np.column_stack(result)

def design(mode, E, D, z, ncontext):
    base = np.tanh(z / 2)
    d = np.clip(E - E.mean(axis=1, keepdims=True), -4, 4)
    features = [d, abs(d), d * base[:, None], np.maximum(d - 1, 0), np.maximum(-d - 1, 0)]
    if mode in ('regime', 'joint'):
        c = D[:, :ncontext]
        features.extend([c, c * base[:, None], (c[:, :, None] * d[:, None, :]).reshape(len(z), -1)])
    if mode in ('evidence', 'joint'):
        e = D[:, ncontext:]
        features.extend([e, np.maximum(e, 0), np.maximum(e - 0.65, 0) * 3, e * base[:, None]])
    return np.column_stack(features)

def fit_head(F, z, y):
    sd = np.maximum(0.25, F.std(0))
    center = F.mean(0)
    X = (F - center) / sd

    def objective(beta):
        adj = z + X @ beta
        loss = np.logaddexp(0, adj).sum() - y @ adj + 0.5 * 700 * (beta @ beta)
        return (loss, X.T @ (expit(adj) - y) + 700 * beta)
    fit = minimize(objective, np.zeros(X.shape[1]), method='L-BFGS-B', jac=True, options={'maxiter': 160, 'ftol': 1e-12})
    if not fit.success:
        raise RuntimeError(fit.message)
    b = fit.x / sd
    return np.r_[b, -center @ b]

def adjust(mode, E, D, z, ncontext, th):
    return z + design(mode, E, D, z, ncontext) @ th[:-1] + th[-1]

def predict_bundle(bundle, output, data=None):
    m = json.loads(bundle.read_text())
    for name, h in m['inputs'].items():
        assert sha(ROOT / name) == h, name
    item, ids, tids, y, wt, X, V, VT, labels, groups, E, D, nc = data or read_data()
    theta = np.array(item['coefficients_and_intercept'])
    z = VT[:, [labels.index(n) for n in item['feature_order']]] @ theta[:-1] + theta[-1]
    result = adjust(m['mode'], E[len(y):], ranks(D[len(y):], np.array(m['quantiles'])), z, nc, np.array(m['weights']))
    return write_submission(expit(result), tids, output)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.replay:
        predict_bundle(args.replay, args.output)
        return
    data = read_data()
    item, ids, tids, y, wt, X, V, VT, labels, groups, E, D, nc = data
    historic = np.load(OLD / 'pruning_oof.npz')
    assert np.array_equal(historic['y'], y)
    results, archive = ({}, {})
    for split in SPLITS:
        pred = {k: np.zeros(len(y)) for k in ['incumbent', *MODES]}
        folders = list(StratifiedKFold(5, shuffle=True, random_state=split).split(X, y))
        for tr, va in folders:
            theta = fit_views(V[tr], y[tr], groups, logistic_fit(X[tr], y[tr]), 100.0)
            train_z = V[tr] @ theta[:-1] + theta[-1]
            val_z = V[va] @ theta[:-1] + theta[-1]
            if split in (7, 29):
                assert np.max(abs(historic[f's{split}_unpruned'][va] - val_z)) < 0.01
                val_z = historic[f's{split}_unpruned'][va]
            pred['incumbent'][va] = val_z
            q = distribution_maps(D[tr])
            Dr, Dv = (ranks(D[tr], q), ranks(D[va], q))
            for mode in MODES:
                th = fit_head(design(mode, E[tr], Dr, train_z, nc), train_z, y[tr])
                pred[mode][va] = adjust(mode, E[va], Dv, val_z, nc, th)
        results[split] = {}
        for mode, p in pred.items():
            metrics = score(y, p, wt)
            metrics['fold_delta'] = [score(y[v], p[v], wt[v])['auc'] - score(y[v], pred['incumbent'][v], wt[v])['auc'] for _, v in folders]
            results[split][mode] = metrics
            archive[f's{split}_{mode}'] = p
        print(split, {k: (round(v['auc'], 7), round(v['weighted_auc'], 7)) for k, v in results[split].items()}, flush=True)
    eligible = [k for k in MODES if all((s[k][m] - s['incumbent'][m] >= 0.0003 for s in results.values() for m in ('auc', 'weighted_auc')))]
    chosen = _rebuild.select('regime', max(eligible, key=lambda k: np.mean([s[k]['auc'] for s in results.values()])) if eligible else None)
    report = _rebuild.annotate({'results': results, 'eligible': eligible, 'chosen': chosen, 'parent_sha256': sha(BASE)})
    OUT.mkdir(exist_ok=True)
    (OUT / 'expert_fusion_results.json').write_text(json.dumps(report, indent=2))
    np.savez(OUT / 'expert_fusion_oof.npz', ids=ids.to_numpy(dtype=str), **archive)
    if chosen:
        dest = PACKAGE / 'expert_fusion_candidate'
        dest.mkdir(exist_ok=True)
        cols = [labels.index(n) for n in item['feature_order']]
        theta = np.array(item['coefficients_and_intercept'])
        z = V[:, cols] @ theta[:-1] + theta[-1]
        tz = VT[:, cols] @ theta[:-1] + theta[-1]
        incumbent = pd.read_csv(PACKAGE / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability'].to_numpy()
        assert np.max(abs(expit(tz) - incumbent)) < 1e-13
        q = distribution_maps(D[:len(y)])
        R = ranks(D, q)
        th = fit_head(design(chosen, E[:len(y)], R[:len(y)], z, nc), z, y)
        files = [BASE, Path(__file__).resolve()]
        files.extend((ROOT / f'{stem}_{part}.parquet' for stem in dict.fromkeys([*CONTEXT, *EVIDENCE]) for part in ['train', 'test']))
        recipe = {'mode': chosen, 'weights': th.tolist(), 'quantiles': q.tolist(), 'inputs': {str(p.relative_to(ROOT)): sha(p) for p in files}, **report}
        path = dest / 'weights.json'
        path.write_text(json.dumps(recipe, indent=2))
        write_submission(expit(adjust(chosen, E[len(y):], R[len(y):], tz, nc, th)), tids, dest / 'submission.csv')
        predict_bundle(path, dest / 'submission_replay.csv', data)
        assert sha(dest / 'submission.csv') == sha(dest / 'submission_replay.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
