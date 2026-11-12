# CPU port; original experiments/weight_space/simple_models.py, SHA256 c04eab69edfd3e155b52b8723687d165f03572a771f6247a752f940ab5500583
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""All-observation baselines, a hand-written likelihood rule, and blinded case cards."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from evaluate import ROOT, OUT, score
from finalize import write_submission
DEST = ROOT / 'new weights/full_series'
LR = ['lr_level_lbf_t', 'lr_scale_lbf_t', 'lr_ar_lr', 'lr_gn_shape_lr_sym', 'lr_bim_shape_lr_sym', 'lr_skew_shape_lr_sym', 'lr_persist']

def series_features(part):
    ids = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
    df = pd.read_parquet(ROOT / f'data/{part}_series.parquet').sort_values(['id', 'time'])
    lengths = df.groupby('id', sort=False).size()
    assert lengths.max() <= 1024 and set(lengths.index) == set(ids)
    aligned = np.zeros((len(ids), 2048), np.float32)
    stats = np.empty((len(ids), 10))
    for name, frame in df.groupby('id', sort=False):
        i = ids.get_loc(name)
        x = frame.value.to_numpy(dtype=float)
        period = frame.period.to_numpy()
        b = int(np.count_nonzero(period == 0))
        assert np.all(period[:b] == 0) and np.all(period[b:] == 1) and (1 < b < len(x) - 1)
        z = (x - x.mean()) / max(x.std(), 1e-08)
        aligned[i, 1024 - b:1024 + len(z) - b] = z
        a, c = (x[:b], x[b:])
        size = np.sqrt(len(a) * len(c) / len(x))

        def phi(v):
            return np.corrcoef(v[:-1], v[1:])[0, 1]
        pa, pc = (phi(a), phi(c))
        rho = np.clip((pa * len(a) + pc * len(c)) / len(x), -0.95, 0.995)
        ea = a[1:] - rho * a[:-1]
        ec = c[1:] - rho * c[:-1]
        sigma = max(np.sqrt((ea.var() * len(ea) + ec.var() * len(ec)) / (len(ea) + len(ec))), 1e-08)
        level = abs(a.mean() - c.mean()) * size * (1 - rho) / sigma
        spread = np.log(max(ec.std(), 1e-08) / max(ea.std(), 1e-08))
        za = (ea - ea.mean()) / max(ea.std(), 1e-08)
        zc = (ec - ec.mean()) / max(ec.std(), 1e-08)
        kurt = max(1.0, (np.mean(np.minimum(za ** 4, 400)) + np.mean(np.minimum(zc ** 4, 400))) / 2 - 1)
        scale = abs(spread) * size * 2 / np.sqrt(kurt)
        ar = abs(pa - pc) * size / np.sqrt(max(1 - rho * rho, 0.04))
        quantiles = np.array([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
        qa, qc = (np.quantile(za, quantiles), np.quantile(zc, quantiles))
        shape = np.mean(np.abs(qc - qa)) * size
        stats[i] = [level, scale, ar, shape, abs(spread), abs(np.log(c.std() / a.std())), abs(np.mean(np.sign(za)) - np.mean(np.sign(zc))) * size, rho, np.log(size), b / len(x)]
    assert np.isfinite(stats).all() and np.isfinite(aligned).all()
    lr = pd.read_parquet(ROOT / f'tune_out/lr5_{part}.parquet', columns=LR).loc[ids]
    lm = pd.read_parquet(ROOT / f'tune_out/lm_{part}.parquet', columns=['lm_z_ar1', 'lm_z_ar2', 'lm_z_ar3']).loc[ids]
    assert lr.index.is_unique and lm.index.is_unique
    return (ids, aligned, stats, lr.to_numpy(), lm.to_numpy(), int(lengths.sum()))

def hand_score(stats):
    level, scale, ar, shape = stats[:, :4].T
    z = np.stack([level, scale, ar, np.maximum(shape - 1.0, 0)], axis=1)
    evidence = 0.43 * np.minimum(z, 7.0) ** 2 - np.array([2.6, 2.3, 2.2, 2.3])
    return expit(logsumexp(evidence + np.log([0.34, 0.34, 0.16, 0.16]), axis=1))

def main():
    OUT.mkdir(exist_ok=True)
    DEST.mkdir(exist_ok=True)
    ids, raw, s, lr, lm, n_train = series_features('train')
    tids, traw, ts, tlr, tlm, n_test = series_features('test')
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    prior = np.load(ROOT / 'tune_out/adv_ptest_train.npy')
    wt = prior / (1 - prior).clip(0.001)
    audit = np.random.default_rng(873123).choice(len(y), 200, replace=False)
    audit_ref = np.ones(len(y), bool)
    audit_ref[audit] = False

    def evidence(summary, likelihood, lm_z):
        return np.column_stack([likelihood[:, 0], likelihood[:, 1], np.maximum(np.sqrt(np.maximum(likelihood[:, 2], 0)), np.abs(lm_z).max(1)), np.max(likelihood[:, 3:6], axis=1), summary[:, 3], summary[:, 7]])
    er, et = (evidence(s, lr, lm), evidence(ts, tlr, tlm))
    assert np.isfinite(er).all() and np.isfinite(et).all()
    quantiles = [0.15, 0.35, 0.55, 0.72, 0.85, 0.93, 0.97, 0.99, 0.997]
    thresholds = np.quantile(er[audit_ref & (y == 0)], quantiles, axis=0)
    cases = np.concatenate([et, er[audit]])
    order = np.random.default_rng(361311).permutation(len(cases))
    cards = (cases[order, None, :] > thresholds[None, :, :]).sum(axis=1)
    lines = [''.join(map(str, c)) for c in cards]
    (OUT / 'case_cards.txt').write_text('\n'.join((' '.join(lines[i:i + 10]) for i in range(0, len(lines), 10))) + '\n')
    np.savez(OUT / 'case_key.npz', order=order, audit=audit, ids=ids.to_numpy(dtype=str), tids=tids.to_numpy(dtype=str))
    scores = {'observations': {'train': n_train, 'test': n_test}}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(s, y))
    saved = {}

    def store(name, oof, test):
        scores[name] = score(y, oof, wt)
        saved[name] = oof
        write_submission(test.astype(float), tids, DEST / f'{name}.csv')
    store('gpt_rule', hand_score(s), hand_score(ts))

    def mechanisms(summary, likelihood, lm_z):
        raw_evidence = np.column_stack([summary[:, :7], likelihood[:, :6], np.abs(lm_z)])
        return np.sign(raw_evidence) * np.log1p(np.abs(raw_evidence))
    for name, X, T, scale, reg in [('raw_points_logistic', raw, traw, False, 0.0003), ('summary_logistic', np.log1p(np.abs(s[:, :7])), np.log1p(np.abs(ts[:, :7])), True, 1.0), ('mechanism_logistic', mechanisms(s, lr, lm), mechanisms(ts, tlr, tlm), True, 1.0)]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=reg, max_iter=400)) if scale else LogisticRegression(C=reg, max_iter=400)
        oof = np.zeros(len(y))
        for tr, va in folds:
            clf.fit(X[tr], y[tr])
            oof[va] = clf.predict_proba(X[va])[:, 1]
        clf.fit(X, y)
        store(name, oof, clf.predict_proba(T)[:, 1])
        estimator = clf[-1] if scale else clf
        np.savez(DEST / f'{name}.npz', weight=estimator.coef_[0], bias=estimator.intercept_[0], center=clf[0].mean_ if scale else 0.0, scale=clf[0].scale_ if scale else 1.0)
        with np.load(DEST / f'{name}.npz') as ck:
            replay = expit((T - ck['center']) / ck['scale'] @ ck['weight'] + ck['bias'])
        assert np.allclose(replay, clf.predict_proba(T)[:, 1], atol=1e-08)
    np.savez(OUT / 'simple_oof.npz', ids=ids.to_numpy(dtype=str), **saved)
    (DEST / 'simple_results.json').write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
