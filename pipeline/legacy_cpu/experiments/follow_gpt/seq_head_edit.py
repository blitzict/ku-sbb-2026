# CPU port of experiments/follow_gpt/seq_head_edit.py, source sha256 69fbf8006bc1306864e0d47fd92b344a43d1d7c837e5609c92f97261a92fb79c. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Last-layer (head.6) refit of the raw-sequence networks (experiments/v6/nn_seq.py), following
experiments/weight_edit_20261104/neural_head_edit.py exactly: frozen encoder, penultimate representation H (input to head.6) averaged
over the 4 TTA views, pairwise logistic refit on the fold's CORE rows (tri of train_fold), lambda in [.03,.3,3] x alpha in [.25,.5,1],
sequential acceptance only if INNER (ini) AUC beats the current best by > 0.0003, intercept reset to preserve original mean core probability.
Outer validation labels are never used for fitting or selection (only reported).
Writes tune_out/seqpart_edit/<tag>_f{f}.npz (va, pv = 4 x n_val logits, pt = 4 x 4000 logits, meta) and edited checkpoints to
weights_experiments/seq_head_edit/<tag>/fold{f}_seed{s}.pt.
Usage (from project root): .venv/Scripts/python -u experiments/follow_gpt/seq_head_edit.py [--resume] [--tags ...] [--folds ...]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import sys, json, time, argparse, ctypes, pathlib, types, warnings; warnings.filterwarnings("ignore")
try:
    k32 = ctypes.windll.kernel32; k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]; k32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
torch.set_num_threads(KU_ENV.threads())
ROOT = KU_ENV.root(); O = ROOT / "tune_out"
OUTD = O / "seqpart_edit"; WD = ROOT / "weights_experiments/seq_head_edit"
dev = "cpu"
RUNS = {"W_SEQ_q": (0, 0), "W_SEQ_q_s1": (0, 1), "W_SEQ_q_s2": (0, 2), "W_SEQ_qh": (8, 0), "W_SEQ_qh_s1": (8, 1)}   # tag -> (hq, seed)
L = 1024; P = 12; CH = "z,e,ae,e2,ind,pos".split(","); AUG = {"rev", "flip", "crop"}; QIN = 16
a = types.SimpleNamespace(qin=QIN, hq=0, bf16=1)   # hq set per run


# ---------------------------------------------------------------- copied verbatim from nn_seq.py (inference path only)
def view(x, n, b, rev, flip):
    t = torch.arange(L, device=dev)[None, :]
    if rev is not None and rev.any():
        idx = (n[:, None] - 1 - t).clamp(min=0); xr = torch.gather(x, 1, idx)
        x = torch.where(rev[:, None], xr, x); b = torch.where(rev, n - b, b)
    x = x * (t < n[:, None])
    if flip is not None: x = torch.where(flip[:, None], -x, x)
    return x, n, b


def mquant(h, m, nq):
    hs = torch.sort(torch.where(m > 0, h, torch.full_like(h, float("inf"))), -1).values; cnt = m.sum(-1, keepdim=True)
    lv = (torch.arange(nq, device=h.device, dtype=h.dtype) + 0.5) / nq; ix = (lv[None, None, :] * (cnt - 1)).round().long()
    return torch.gather(hs, -1, ix.expand(-1, h.shape[1], -1))


def channels(x, n, b):
    Bt = x.shape[0]; Lb = int(((int(n.max()) + 63) // 64) * 64); x = x[:, :Lb]; t = torch.arange(Lb, device=dev)[None, :]
    valid = (t < n[:, None]).float(); bef = (t < b[:, None]).float(); aft = valid - bef
    nb = bef.sum(1, keepdim=True); m = (x * bef).sum(1, keepdim=True) / nb; sd = (((x - m) ** 2 * bef).sum(1, keepdim=True) / nb).sqrt().clamp(min=1e-6)
    z = ((x - m) / sd).clamp(-10, 10) * valid
    lags = torch.stack([F.pad(z, (k, 0))[:, :Lb] for k in range(1, P + 1)] + [torch.ones_like(z)], -1)
    rw = (bef * (t >= P).float()); mrow = rw.sum(1)
    Xw = lags * rw[..., None]; XtX = Xw.transpose(1, 2) @ lags; Xty = (Xw * z[..., None]).sum(1); yy = (z * z * rw).sum(1)
    best_bic = torch.full((Bt,), float("inf"), device=dev); beta = torch.zeros(Bt, P + 1, device=dev)
    for k in range(P + 1):
        sel = list(range(k)) + [P]; A = XtX[:, sel][:, :, sel] + 1e-4 * torch.eye(k + 1, device=dev); c = Xty[:, sel]
        bk = torch.linalg.solve(A, c[..., None])[..., 0]; rss = (yy - (bk * c).sum(1)).clamp(min=1e-6)
        bic = mrow * torch.log(rss / mrow) + (k + 1) * torch.log(mrow); better = bic < best_bic
        full = torch.zeros(Bt, P + 1, device=dev); full[:, sel] = bk
        beta = torch.where(better[:, None], full, beta); best_bic = torch.where(better, bic, best_bic)
    e = (z - (lags * beta[:, None, :]).sum(-1)) * valid * (t >= P).float()
    es = ((e * e * rw).sum(1, keepdim=True) / mrow[:, None]).sqrt().clamp(min=1e-6); e = (e / es).clamp(-10, 10)
    feats = {"z": z, "e": e, "ae": e.abs(), "e2": (e * e).clamp(max=25) / 5, "ind": aft, "pos": t / n[:, None].float() * valid}
    xc = torch.stack([feats[c] for c in CH], 1)
    q = torch.cat([mquant(torch.stack([z, e], 1), mm[:, None, :], a.qin).flatten(1) for mm in (bef, aft)], 1); return xc, bef, aft, q


class Block(nn.Module):
    def __init__(s, c, d, k, p):
        super().__init__(); s.n1 = nn.BatchNorm1d(c); s.c1 = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d)
        s.n2 = nn.BatchNorm1d(c); s.c2 = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d); s.dp = nn.Dropout(p)
    def forward(s, h, m): return (h + s.c2(s.dp(F.gelu(s.n2(s.c1(F.gelu(s.n1(h)))))))) * m


class Net(nn.Module):
    def __init__(s, cin=6, c=64, dil=(1, 2, 4, 8, 16, 32, 1, 4), k=5, drop=0.3, naux=0):
        super().__init__(); nq = 2 * 2 * a.qin; s.qn = nn.Linear(3 * nq // 2, 64)
        npool = (2 + (a.hq if a.hq else 1)) * c; s.stem = nn.Conv1d(cin, c, 7, padding=3); s.blocks = nn.ModuleList([Block(c, d, k, 0.1) for d in dil]); s.nf = nn.BatchNorm1d(c)
        s.head = nn.Sequential(nn.Linear(4 * npool + 2 + naux + 64, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop / 2), nn.Linear(64, 1))
    @staticmethod
    def pool(h, m):
        cnt = m.sum(-1, keepdim=True).clamp(min=1); mu = (h * m).sum(-1, keepdim=True) / cnt
        sd = (((h - mu) ** 2 * m).sum(-1) / cnt[..., 0]).clamp(min=1e-6).sqrt()
        mx = mquant(h, m, a.hq).flatten(1) if a.hq else (h + (m - 1) * 1e4).amax(-1)
        return torch.cat([mu[..., 0], sd, mx], 1)
    def forward(s, xc, bef, aft, aux, q=None):
        v = (bef + aft)[:, None, :]; h = s.stem(xc) * v
        for blk in s.blocks: h = blk(h, v)
        h = F.gelu(s.nf(h)).float(); pb, pa = s.pool(h, bef[:, None, :]), s.pool(h, aft[:, None, :])
        ln = torch.stack([bef.sum(1).log() - 6, aft.sum(1).log() - 6], 1)
        qb, qa = q.float().chunk(2, 1); extra = [F.gelu(s.qn(torch.cat([qb, qa, qa - qb], 1)))]
        return s.head(torch.cat([pb, pa, pa - pb, (pa - pb).abs(), ln, aux] + extra, 1))[:, 0]


CAP = []
def run(model, x, n, b, aux, rev, flip):
    xv, nv, bv = view(x, n, b, rev, flip); xc, bef, aft, q = channels(xv, nv, bv)
    with torch.autocast("cpu", enabled=False): return model(xc, bef, aft, aux, q).float()


@torch.no_grad()
def predict(model, x, n, b, aux, bs=512):
    """nn_seq.predict, also returning the head.6 input per view: logits (4, rows), H (4, rows, 64), original order."""
    model.eval(); out, hs = [], []; srt = torch.argsort(n); inv = torch.argsort(srt).cpu().numpy(); x, n, b, aux = x[srt], n[srt], b[srt], aux[srt]
    for i in range(0, len(n), bs):
        sl = slice(i, i + bs); r, hh = [], []
        for rv in (False, True):
            for fl in (False, True):
                k = n[sl].shape[0]; CAP.clear(); r.append(run(model, x[sl], n[sl], b[sl], aux[sl], torch.full((k,), rv, device=dev), torch.full((k,), fl, device=dev)))
                hh.append(CAP[0])
        out.append(torch.stack(r, 0)); hs.append(torch.stack(hh, 0))
    return torch.cat(out, 1).cpu().numpy()[:, inv], torch.cat(hs, 1)[:, torch.as_tensor(inv, device=dev)]


# ---------------------------------------------------------------- edit rule, as neural_head_edit.edit_head (H = TTA-mean representation)
def edit_head(H, y, original, core, inner, seed):
    original_w, original_b = original
    htr = H[torch.as_tensor(core, device=dev)]; scale = htr.std(0).clamp(min=.03); x = htr / scale; b0 = htr @ original_w + original_b
    hin = H[torch.as_tensor(inner, device=dev)]; base_inner = (hin @ original_w + original_b).cpu().numpy(); base_auc = roc_auc_score(y[inner], base_inner)
    best_auc, best_alpha, best_lam = base_auc, 0., None; best_delta = np.zeros(H.shape[1], np.float64)
    rng = np.random.default_rng(seed)
    pos = torch.as_tensor(rng.choice(np.flatnonzero(y[core] == 1), 262144), device=dev); neg = torch.as_tensor(rng.choice(np.flatnonzero(y[core] == 0), 262144), device=dev)
    trials = []
    for lam in (.03, .3, 3.):
        def objective(delta):
            d = torch.tensor(delta, device=dev, dtype=torch.float32, requires_grad=True); z = b0 + x @ d
            loss = F.softplus(-(z[pos] - z[neg])).mean() + lam * (d * d).sum(); loss.backward()
            return float(loss.detach().cpu()), d.grad.detach().cpu().numpy().astype(np.float64)
        fit = minimize(objective, np.zeros(H.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 160, "gtol": 2e-6, "ftol": 2e-9, "maxls": 30})
        delta = torch.as_tensor(fit.x, dtype=torch.float32, device=dev) / scale; change = (hin @ delta).cpu().numpy()
        for alpha in (.25, .5, 1.):
            auc = float(roc_auc_score(y[inner], base_inner + alpha * change))
            trials.append({"lambda": lam, "alpha": alpha, "inner_auc": auc, "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message)})
            if fit.success and auc > best_auc + .0003: best_auc, best_alpha, best_lam = auc, alpha, lam; best_delta = fit.x.copy()
    new_w = original_w + best_alpha * torch.as_tensor(best_delta, dtype=torch.float32, device=dev) / scale
    train_z = (htr @ new_w).cpu().numpy(); original_mean = float(torch.sigmoid(b0).mean().cpu()); bias = float(original_b.cpu())
    if best_alpha:
        for _ in range(20):
            p = expit(train_z + bias); step = (p - original_mean).sum() / max(float((p * (1 - p)).sum()), 1e-8)
            bias -= float(np.clip(step, -2, 2))
            if abs(step) < 1e-8: break
    return new_w, bias, {"inner_baseline": float(base_auc), "inner_selected": float(best_auc), "alpha": best_alpha, "lambda": best_lam, "trials": trials}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*", default=list(RUNS)); ap.add_argument("--folds", type=int, nargs="*", default=list(range(5)))
    ap.add_argument("--resume", action="store_true"); args = ap.parse_args(); t0 = time.time()
    OUTD.mkdir(parents=True, exist_ok=True)
    ids = pd.read_parquet(ROOT / "clean/features_train.parquet", columns=[]).index; tids = pd.read_parquet(ROOT / "clean/features_test.parquet", columns=[]).index
    ztr = np.load(O / "seq_cache_train.npz", allow_pickle=True); zte = np.load(O / "seq_cache_test.npz", allow_pickle=True)
    assert list(ztr["ids"]) == list(ids) and list(zte["ids"]) == list(tids)
    y = pd.read_csv(ROOT / "data/train_labels.csv", dtype={"id": str}).set_index("id").label.loc[ids].to_numpy().astype(np.float32)
    gX, gN, gB = [torch.tensor(ztr[k], device=dev) for k in ("x", "n", "b")]; tX, tN, tB = [torch.tensor(zte[k], device=dev) for k in ("x", "n", "b")]
    gA = torch.zeros((len(ids), 0), device=dev); tA = torch.zeros((len(tids), 0), device=dev)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
    print(f"loaded {gX.shape} {tX.shape} on {"cpu"} ({time.time()-t0:.0f}s)", flush=True)
    for tag in args.tags:
        hq, sd = RUNS[tag]; a.hq = hq
        for f in args.folds:
            part = OUTD / f"{tag}_f{f}.npz"
            if args.resume and part.exists(): print(f"reuse {part.name}", flush=True); continue
            tr, va = folds[f]; seed = 1000 * f + sd
            ref = np.load(O / f"seqpart/{tag}_f{f}.npz", allow_pickle=True); assert np.array_equal(ref["va"], va)
            rm = json.loads(str(ref["meta"])); assert len(rm) == 1 and rm[0]["seed"] == sd and rm[0]["fold"] == f
            tri, ini = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=seed)
            assert not set(tri) & set(va) and not set(ini) & set(va) and not set(tri) & set(ini)
            ck = ROOT / "weights" / tag / f"fold{f}_seed{sd}.pt"; state = torch.load(ck, map_location="cpu", weights_only=True)
            model = Net().to(dev); model.load_state_dict(state, strict=True); model.eval()
            for p_ in model.parameters(): p_.requires_grad_(False)
            hk = model.head[6].register_forward_pre_hook(lambda mod, inp: CAP.append(inp[0].float()))
            t = lambda idx: torch.as_tensor(idx, device=dev)
            pv0, Hva = predict(model, gX[t(va)], gN[t(va)], gB[t(va)], gA[t(va)]); pt0, Hte = predict(model, tX, tN, tB, tA)
            pin, Hin = predict(model, gX[t(ini)], gN[t(ini)], gB[t(ini)], gA[t(ini)]); _, Hco = predict(model, gX[t(tri)], gN[t(tri)], gB[t(tri)], gA[t(tri)])
            hk.remove()
            ev, et = np.abs(pv0 - ref["pv"]), np.abs(pt0 - ref["pt"])
            inner_replay = roc_auc_score(y[ini], pin.mean(0))
            print(f"{tag} f{f}: replay logit err val max {ev.max():.3g} mean {ev.mean():.3g}  test max {et.max():.3g} mean {et.mean():.3g}; inner auc {inner_replay:.5f} vs logged {rm[0]['inner']:.5f}", flush=True)
            assert ev.max() < 0.05 and et.max() < 0.05 and ev.mean() < 2e-3 and abs(inner_replay - rm[0]["inner"]) < 2e-4, "replay mismatch - stop"
            # TTA-mean representation over all rows (core, inner); head is linear so mean-logit = head(mean H)
            H = torch.zeros((len(ids), 64), device=dev); H[t(tri)] = Hco.mean(0); H[t(ini)] = Hin.mean(0)
            w0 = model.head[6].weight.detach().flatten().float(); b0 = model.head[6].bias.detach().flatten()[0].float()
            wnew, bnew, info = edit_head(H, y, (w0, b0), tri, ini, 20261104 + 100 * f + 10 * list(RUNS).index(tag) + sd)
            dw = wnew - w0; db = bnew - float(b0.cpu())
            # edited per-view logits = original per-view logits + linear change (exactly identical when no edit accepted)
            pv = ref["pv"] + (Hva @ dw).cpu().numpy() + db if info["alpha"] else ref["pv"].copy()
            pt = ref["pt"] + (Hte @ dw).cpu().numpy() + db if info["alpha"] else ref["pt"].copy()
            new_state = {k: v.clone() for k, v in state.items()}; new_state["head.6.weight"] = wnew.detach().cpu()[None]; new_state["head.6.bias"] = torch.tensor([bnew])
            assert all(torch.equal(new_state[k], state[k]) for k in state if k not in ("head.6.weight", "head.6.bias"))
            (WD / tag).mkdir(parents=True, exist_ok=True); torch.save(new_state, WD / tag / f"fold{f}_seed{sd}.pt")
            info.update(fold=f, seed=sd, tag=tag, replay_val_max=float(ev.max()), replay_test_max=float(et.max()), weight_delta_l2=float(torch.linalg.vector_norm(dw).cpu()),
                        outer_baseline_report_only=float(roc_auc_score(y[va], ref["pv"].mean(0))), outer_edited_report_only=float(roc_auc_score(y[va], pv.mean(0))))
            np.savez(part, va=va, pv=pv.astype(np.float32), pt=pt.astype(np.float32), meta=json.dumps(rm), edit=json.dumps(info))
            print(f"{tag} f{f}: inner {info['inner_baseline']:.5f}->{info['inner_selected']:.5f} lam={info['lambda']} alpha={info['alpha']}  "
                  f"outer(report) {info['outer_baseline_report_only']:.5f}->{info['outer_edited_report_only']:.5f}  ({time.time()-t0:.0f}s)", flush=True)
            del model, Hva, Hte, Hin, Hco, H; pass
    print(f"done {time.time()-t0:.0f}s", flush=True)


def evaluate():
    """CPU-only: SEQ bag alone and the 23-column stack, original vs edited SEQ views (run from project root)."""
    os.chdir(ROOT); sys.path.insert(0, str(KU_ENV.experiments() / "follow_gpt")); import common as C
    ed = [json.loads(str(np.load(OUTD / f"{t}_f{f}.npz", allow_pickle=True)["edit"])) for t in RUNS for f in range(5)]
    print(f"heads changed {sum(e['alpha'] > 0 for e in ed)}/25: " + ", ".join(f"{e['tag']} f{e['fold']} (lam {e['lambda']}, a {e['alpha']})" for e in ed if e["alpha"]))
    print(f"max replay logit error val {max(e['replay_val_max'] for e in ed):.3g} test {max(e['replay_test_max'] for e in ed):.3g}")
    for sub in ("seqpart", "seqpart_edit"):
        bag = np.mean(C.seq_views(C.SEQ5, "oof", sub), 0)
        print(f"SEQ5 bag alone [{sub}]: {roc_auc_score(C.y, bag):.5f} / {roc_auc_score(C.y, bag, sample_weight=C.w):.5f}")
    res = {}
    for nm, X in [("ref views (seqpart)", C.gpt_design(seq="views")), ("edited views (seqpart_edit)", C.gpt_design(seq="views", seqdir="seqpart_edit"))]:
        res[nm] = C.report(nm, X)
    json.dump(dict(edits=ed, stack=res), open(OUTD / "results.json", "w"), indent=1)


if __name__ == "__main__":
    if "--eval" in sys.argv: evaluate()
    else: main()
