"""Saved-network prediction for nn_var, nn_xpu, nn_seq, neural_head_edit and replay_head_edits.

Networks, predict loops and per-series preprocessing are taken from the training source. Rank-Gaussian inputs use the
saved per-fold sorted training reference (KU_ENV.rank_reference / rank_gauss_bounded format) and the same
rank_gauss_bounded.transform the training used; nothing is refitted. neural_head_edit and replay_head_edits rebuild
their reference from the source nn_var rows in training; those rows are exactly the rows of nn_var's saved
fold<k>_rankgauss_ref.npz (same core permutation, shift rows, versions and extra rows), so that file is used.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit, logit

from pipeline.inference import features as FE
from pipeline.inference import source

import rank_gauss_bounded as RB  # noqa: E402  (pipeline/legacy_cpu on sys.path via source)

sig = lambda z: 1 / (1 + np.exp(-z))


def reference(path):
    """(srt, nv) exactly as rank_gauss_bounded.sorted_reference built them, from the saved sorted training rows."""
    with np.load(source.require(path)) as z:
        ordered = z["sorted"]
    t = torch.from_numpy(np.ascontiguousarray(ordered.T))
    t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
    return t, torch.isfinite(t).sum(1, keepdim=True).clamp(min=1).float()


def _state(path):
    state = torch.load(source.require(path), map_location="cpu", weights_only=True)
    if not all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError(f"non-finite weights in {path}")
    return state


def _model(cls, state, *args):
    m = cls(*args)
    m.load_state_dict(state)
    m.eval()
    return m


def _versions(feature_root, cols, kinds):
    return [FE.var_test(feature_root, cols).to_numpy()] + [FE.var_test(feature_root, cols, k).to_numpy() for k in kinds]


def nn_var(stage, model_root, feature_root):
    script = stage["argv"][0]
    a = source.parse_args(script, stage["argv"][1:])
    torch.set_num_threads(source.ku_workspace.threads())
    ns = source.definitions(script, ["PLR", "MLP", "predict"], {"a": a, "nn": nn, "torch": torch, "np": np, "dev": "cpu"})
    kinds = tuple(a.kinds.split(","))
    cols = FE.matrix_columns(model_root, Path(a.mat).stem)
    T = _versions(feature_root, cols, kinds)
    n, NV = len(T[0]), 1 + len(kinds)
    wd = Path(model_root) / "weights" / a.tag
    tp, ttp, vert = np.zeros(n), np.zeros(n), np.zeros((NV, n))
    for k in range(5):
        srt, nv = reference(wd / f"fold{k}_rankgauss_ref.npz")
        Zt = [RB.transform(x, srt, nv) for x in T]
        pt = np.zeros((NV, n))
        for s in range(a.seeds):
            model = _model(ns["MLP"], _state(wd / f"fold{k}_seed{s}.pt"), len(cols))
            for j in range(NV):
                pt[j] += ns["predict"](model, Zt[j]) / a.seeds
        tp += pt[0] / 5; ttp += pt.mean(0) / 5; vert += pt / 5
    return {f"test_{a.tag}": sig(tp), f"test_{a.tag}_tta": sig(ttp),
            f"ver_{a.tag}": {"test": sig(vert).astype(np.float32), "kinds": np.array(["orig"] + list(kinds))}}


def nn_xpu(stage, model_root, feature_root):
    script = stage["argv"][0]
    a = source.parse_args(script, stage["argv"][1:])
    if a.no_test:
        raise ValueError(f"{a.tag}: trained with --no-test")
    torch.set_num_threads(source.ku_workspace.threads())
    ns = source.definitions(script, ["MLP", "predict"], {"nn": nn, "torch": torch, "np": np, "dev": "cpu"})
    blocks = tuple(b for b in a.blocks.split(",") if b)
    names = FE.featset_columns(model_root, blocks, rd=a.rd)
    if a.cols:
        names = [c for c in pd.read_csv(source.require(Path(model_root) / a.cols)).feature if c in set(names)]
    Xt = FE.featset_test(model_root, feature_root, blocks, rd=a.rd)
    missing = [c for c in names if c not in Xt.columns]
    if missing:
        raise KeyError(f"{a.tag}: trained columns missing from new features, e.g. {missing[:3]}")
    Xtn = Xt[names].to_numpy()
    wd = Path(model_root) / "weights" / a.tag
    test = np.zeros(len(Xtn))
    for k in range(5):
        pt = np.zeros(len(Xtn))
        for s in range(a.seeds):
            seed = 100 * k + s
            srt, nv = reference(wd / f"seed{seed}_rankgauss_ref.npz")
            Z = RB.transform(Xtn, srt, nv)
            model = _model(ns["MLP"], _state(wd / f"seed{seed}.pt"), len(names), a.hidden, a.layers, a.drop)
            pt += ns["predict"](model, Z) / a.seeds
        test += pt / 5
    return {f"test_{a.tag}": sig(test)}


def nn_seq(stage, model_root, feature_root):
    script = stage["argv"][0]
    a = source.parse_args(script, stage["argv"][1:])
    if [c for c in a.aux.split(",") if c]:
        raise ValueError("--aux rank-transforms with an unsaved training reference; not part of the recipe")
    if a.folds and sorted(a.folds) != list(range(5)):
        raise ValueError("partial-fold nn_seq runs have no complete test prediction")
    torch.set_num_threads(a.threads)
    feature_root = Path(feature_root)
    ns = {"a": a, "dev": "cpu", "L": 1024, "P": a.P, "AUG": set(x for x in a.aug.split(",") if x), "CH": a.chans.split(","),
          "torch": torch, "nn": nn, "F": F, "math": math, "np": np, "pd": pd, "D": feature_root, "O": feature_root / "tune_out"}
    source.definitions(script, ["load", "view", "channels", "mquant", "Block", "Net", "run", "predict"], ns)
    tids, XT, NT, BT = ns["load"]("test")   # per-series cache built from work/data/test_series.parquet when absent
    tX, tN, tB = (torch.tensor(v) for v in (XT, NT, BT))
    tA = torch.zeros((len(tids), 0), dtype=torch.float32)
    wd = Path(model_root) / "weights" / a.tag
    tst = []
    for f in range(5):
        tl = []
        for sd in a.seeds:
            model = _model(ns["Net"], _state(wd / f"fold{f}_seed{sd}.pt"), len(ns["CH"]), a.ch, [int(d) for d in a.dil.split(",")], a.k, a.drop, 0)
            tl.append(ns["predict"](model, tX, tN, tB, tA))
        tst.append(np.mean(tl, 0).mean(0))
    return {f"test_{a.tag}": sig(np.mean(tst, 0))}


HEAD = "experiments/weight_edit_20261104/neural_head_edit.py"
HEAD_DIR = "weights_experiments/weight_edit_20261104/neural_head"


def _head_parts():
    return source.definitions(HEAD, ["KINDS", "VIEW_WEIGHTS", "MLP", "transform"], {"np": np, "nn": nn, "torch": torch})


def neural_head_edit(stage, model_root, feature_root):
    a = _head_args(stage["argv"][1:])
    torch.set_num_threads(source.ku_workspace.threads())
    ns = _head_parts()
    KINDS, VW = ns["KINDS"], ns["VIEW_WEIGHTS"]
    cols = FE.matrix_columns(model_root, "mat_cggd_hv")
    T = dict(zip(KINDS, _versions(feature_root, cols, KINDS[1:])))
    n = len(T["orig"])
    model_root = Path(model_root)
    out, ot, et = {}, [], []
    for fold in range(5):
        srt, nv = reference(model_root / "weights" / a.source / f"fold{fold}_rankgauss_ref.npz")
        models = [_model(ns["MLP"], _state(model_root / "weights" / a.source / f"fold{fold}_seed{s}.pt"), len(cols)) for s in range(4)]
        hidden = {s: torch.zeros((n, 1024)) for s in a.edit_seeds}
        for j, kind in enumerate(KINDS):
            data = T[kind]
            for start in range(0, n, 8192):
                end = min(n, start + 8192)
                batch = ns["transform"](data[start:end], srt, nv)
                with torch.no_grad(), torch.autocast("cpu", enabled=False):
                    for s, model in enumerate(models):
                        if s in hidden:
                            hidden[s][start:end] += float(VW[j]) * model.net[:-1](batch).float()
        original, edited = [], []
        for s in a.edit_seeds:
            w0 = models[s].net[-1].weight.detach().flatten().float()
            b0 = models[s].net[-1].bias.detach().flatten()[0].float()
            new = _state(model_root / HEAD_DIR / a.run_name / f"fold{fold}_seed{s}.pt")
            original.append((hidden[s] @ w0 + b0).numpy())
            edited.append((hidden[s] @ new["net.12.weight"][0] + float(new["net.12.bias"][0])).numpy())
        ot.append(np.stack(original)); et.append(np.stack(edited))
        out[f"weight_edit_20261104/{a.run_name}/head_fold{fold}"] = {"original_test": ot[-1], "edited_test": et[-1]}
    tids = np.asarray(FE.test_ids(feature_root)).astype(str)
    out[f"weight_edit_20261104/{a.run_name}/head_oof"] = {"original_test": np.mean(ot, axis=0), "edited_test": np.mean(et, axis=0), "test_ids": tids}
    return out


def _head_args(argv):
    """neural_head_edit builds its parser inside main(); these are its options and defaults."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--source", default="MLPHV_big")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--run-name", default="head_v2_MLPHV_big")
    ap.add_argument("--edit-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    a = ap.parse_args(argv)
    if sorted(a.folds) != list(range(5)):
        raise ValueError("partial-fold head edits have no complete test prediction")
    return a


def replay_head_edits(stage, model_root, feature_root):
    ns = _head_parts()
    KINDS, VW = ns["KINDS"], ns["VIEW_WEIGHTS"]
    torch.set_num_threads(source.ku_workspace.threads())
    cols = FE.matrix_columns(model_root, "mat_cggd_hv")
    T = dict(zip(KINDS, _versions(feature_root, cols, KINDS[1:])))
    n = len(T["orig"])
    model_root = Path(model_root)
    checkpoint_root = model_root / (stage["argv"][stage["argv"].index("--checkpoint-root") + 1] if "--checkpoint-root" in stage["argv"] else HEAD_DIR)
    out, tag_tests = {}, []
    for source_tag in ["MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2"]:
        run = f"head_v2_{source_tag}"
        testv = np.zeros((6, n))
        for fold in range(5):
            srt, nv = reference(model_root / "weights" / source_tag / f"fold{fold}_rankgauss_ref.npz")
            models = [_model(ns["MLP"], _state(checkpoint_root / run / f"fold{fold}_seed{s}.pt"), len(cols)) for s in range(4)]
            per_seed = np.zeros((4, 6, n))
            for j, kind in enumerate(KINDS):
                data = T[kind]
                for start in range(0, n, 8192):
                    end = min(start + 8192, n)
                    batch = ns["transform"](data[start:end], srt, nv)
                    with torch.no_grad():
                        for s, model in enumerate(models):
                            with torch.autocast("cpu", enabled=False):
                                h = model.net[:-1](batch).float()
                            per_seed[s, j, start:end] = model.net[-1](h).squeeze(-1).numpy()
            testv += per_seed.mean(0) / 5
        tp = expit(testv).astype(np.float32)
        out[f"weight_edit_20261104/ver_{run}"] = {"test": tp, "kinds": np.array(KINDS)}
        tag_tests.append(VW @ logit(np.clip(tp.astype(float), 1e-6, 1 - 1e-6)))
    out["weight_edit_20261104/test_head_v2_bag"] = expit(np.mean(tag_tests, axis=0))
    return out
