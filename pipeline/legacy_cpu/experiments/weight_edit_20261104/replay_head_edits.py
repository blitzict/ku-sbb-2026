# CPU port of experiments/weight_edit_20261104/replay_head_edits.py, source sha256 08bf098fe91e34a28633bed48ead49b7228cab2b8d5a9eeb83f5989d17f72e3b. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Reload edited checkpoints and export predictions using the original TTA recipe.

Encoder uses its original BF16 inference; the edited final linear layer uses
float32, matching the head optimization. All seed/fold/view aggregation and
clipping otherwise follow nn_var.py and wtta.py exactly.
"""
import ctypes
import argparse
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from neural_head_edit import ROOT, OUT, WD, KINDS, VIEW_WEIGHTS, MLP, rank_reference, transform, report, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", default=str(WD), help="Edited checkpoint root; may be 'new weights/neural_head'")
    args = ap.parse_args()
    checkpoint_root = ROOT / args.checkpoint_root
    KU_ENV.low_priority()
    torch.set_num_threads(KU_ENV.threads())
    started = time.time()
    m = np.load(ROOT / "tune_out/var/mat_cggd_hv.npz", allow_pickle=True)
    X = {"orig": m["X"], **{k: m[f"V_{k}"] for k in KINDS[1:]}}
    T = {"orig": m["Xt"], **{k: m[f"VT_{k}"] for k in KINDS[1:]}}
    A, E, y, ids, tids, wt = (m[k] for k in ["A", "E", "y", "ids", "tid", "w"])
    source_rows = pd.Index(ids).get_indexer(m["src"])
    extra_rows = pd.Index(ids).get_indexer(m["Esrc"])
    assert (source_rows >= 0).all() and (extra_rows >= 0).all()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X["orig"], y))
    tag_scores = []
    tag_tests = []
    comparisons = {}
    max_errors = []
    for offset, source in enumerate(["MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2"]):
        run = f"head_v2_{source}"
        oofv = np.zeros((6, len(y)))
        testv = np.zeros((6, len(tids)))
        for fold, (tr, va) in enumerate(folds):
            perm = np.random.RandomState(fold + 100 * offset).permutation(tr)
            core = perm[len(tr) // 10:]
            is_core = np.zeros(len(y), bool)
            is_core[core] = True
            rows = np.vstack([X["orig"][core], A[is_core[source_rows]], *[X[k][core] for k in KINDS[1:]], E[is_core[extra_rows]]]).astype(np.float32)
            srt, nv = rank_reference(rows)
            del rows
            models = []
            md = json.loads((checkpoint_root / run / f"fold{fold}.json").read_text())
            for seed in range(4):
                origpath = ROOT / "weights" / source / f"fold{fold}_seed{seed}.pt"
                path = checkpoint_root / run / f"fold{fold}_seed{seed}.pt"
                assert digest(origpath) == md["edits"][seed]["source_sha256"]
                original = torch.load(origpath, weights_only=True, map_location="cpu")
                edited = torch.load(path, weights_only=True, map_location="cpu")
                assert all(torch.isfinite(v).all() for v in edited.values())
                assert all(torch.equal(original[k], edited[k]) for k in original if k not in ["net.12.weight", "net.12.bias"])
                model = MLP(X["orig"].shape[1]).to("cpu")
                model.load_state_dict(edited)
                model.eval()
                models.append(model)
            per_seed = np.zeros((4, 6, len(va) + len(tids)))
            for j, kind in enumerate(KINDS):
                for off, data in [(0, X[kind][va]), (len(va), T[kind])]:
                    for start in range(0, len(data), 8192):
                        end = min(start + 8192, len(data))
                        batch = transform(data[start:end], srt, nv)
                        with torch.no_grad():
                            for seed, model in enumerate(models):
                                with torch.autocast("cpu", enabled=False):
                                    h = model.net[:-1](batch).float()
                                pred = model.net[-1](h).squeeze(-1)
                                per_seed[seed, j, off + start:off + end] = pred.cpu().numpy()
            reference = np.load(OUT / run / f"head_fold{fold}.npz", allow_pickle=False)
            check = np.concatenate([reference["edited"], reference["edited_test"]], axis=1)
            replay = np.einsum("j,sjn->sn", VIEW_WEIGHTS, per_seed)
            max_error = float(np.max(np.abs(replay - check)))
            assert max_error < 1e-4, f"Serialized inference mismatch {max_error}"
            max_errors.append(max_error)
            oofv[:, va] = per_seed[:, :, :len(va)].mean(0)
            testv += per_seed[:, :, len(va):].mean(0) / 5
            del models, srt, nv
            gc.collect()
            pass
            print(f"Replay {source} fold {fold}: unchanged encoder verified, head max error {max_error:.3g} ({time.time()-started:.0f}s)", flush=True)
        # Mirror the archive's float32 probabilities and subsequent logit clamp.
        op, tp = expit(oofv).astype(np.float32), expit(testv).astype(np.float32)
        np.savez(OUT / f"ver_{run}.npz", oof=op, test=tp, kinds=np.array(KINDS))
        s = VIEW_WEIGHTS @ logit(np.clip(op.astype(float), 1e-6, 1-1e-6))
        t = VIEW_WEIGHTS @ logit(np.clip(tp.astype(float), 1e-6, 1-1e-6))
        tag_scores.append(s)
        tag_tests.append(t)
        old = np.load(ROOT / "tune_out" / f"ver_{source}.npz", allow_pickle=False)
        old_s = VIEW_WEIGHTS @ logit(np.clip(old["oof"].astype(float), 1e-6, 1-1e-6))
        comparisons[source] = {"original": report(y, old_s, wt), "edited": report(y, s, wt)}
        print(json.dumps({source: comparisons[source]}), flush=True)
    new_oof = expit(np.mean(tag_scores, axis=0))
    new_test = expit(np.mean(tag_tests, axis=0))
    old_oof = np.load(ROOT / "tune_out/oof_MLPHV_bag_w.npy", allow_pickle=False)
    comparisons["full_12_seed_bag"] = {"original": report(y, old_oof, wt), "edited": report(y, new_oof, wt)}
    comparisons["max_serialized_replay_logit_error"] = max(max_errors)
    comparisons["checkpoint_count"] = 60
    comparisons["validation"] = "Canonical base-model outer folds; edit strength selected inside original training fold"
    np.save(OUT / "oof_head_v2_bag.npy", new_oof)
    np.save(OUT / "test_head_v2_bag.npy", new_test)
    (OUT / "head_replay_results.json").write_text(json.dumps(comparisons, indent=2))
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
