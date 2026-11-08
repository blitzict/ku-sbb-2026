# CPU port of experiments/v6/nn_seq.py, source sha256 e69d1229efe8e0b48cbd85c0d13e64631adffbc59ad1fadcca4a61e18a591682. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Raw-sequence before-vs-after network (CPU port), trained from scratch on the real series only.
Per batch, on the device: label-preserving augmentation of the real series (outer-end crops, time reversal with boundary n-b, sign flip), then
channels: z (before mean/sd), AR(p<=12, BIC) residuals fitted on the before part by least squares and scaled by the before residual sd, |e|, e^2,
boundary indicator, relative position. Encoder: dilated 1-D ResNet over the whole series; masked mean/std/max pooling separately over the before
and after parts; head on [before, after, after-before, |after-before|, log lengths (, optional aux scalars)].
CV: StratifiedKFold(5, shuffle, random_state=0) on the clean index; early stopping on an inner 10% split of the training fold only.
TTA: mean logit over orig / flip / rev / revflip. Test = mean of fold models.
Usage: .venv/Scripts/python nn_seq.py --tag SEQ_x [--folds 0] [--aux lm_lmsum,lm_lbf_mix] [--seeds 0]
Writes tune_out/oof_<tag>.npy, test_<tag>.npy, best_<tag>.json (only when all 5 folds run)."""
import sys, json, time, argparse, pathlib, warnings, ctypes, math; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
try:   # explicit 64-bit handle types: the bare call truncates the pseudo-handle and silently does nothing
    k32 = ctypes.windll.kernel32; k32.GetCurrentProcess.restype = ctypes.c_void_p; k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.special import ndtri
D = KU_ENV.root(); O = D / "tune_out"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--folds", type=int, nargs="*")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=8); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--ch", type=int, default=64)
ap.add_argument("--dil", default="1,2,4,8,16,32,1,4"); ap.add_argument("--k", type=int, default=5); ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--P", type=int, default=12); ap.add_argument("--aug", default="rev,flip,crop"); ap.add_argument("--crop-p", type=float, default=0.5)
ap.add_argument("--crop-max", type=float, default=0.4); ap.add_argument("--aux", default=""); ap.add_argument("--seeds", type=int, nargs="*", default=[0])
ap.add_argument("--chans", default="z,e,ae,e2,ind,pos"); ap.add_argument("--bf16", type=int, default=1); ap.add_argument("--threads", type=int, default=KU_ENV.threads())
ap.add_argument("--extra", default="", help="extra real-segment rows (var stems, e.g. hv) added to each training fold by source id"); ap.add_argument("--qin", type=int, default=0, help="add masked quantiles (this many levels) of input z and e per side to the head"); ap.add_argument("--hq", type=int, default=0, help="quantile pooling (levels) of encoder output instead of max")
a = ap.parse_args(); t0 = time.time(); torch.set_num_threads(a.threads)
dev = "cpu"
L = 1024; P = a.P; AUG = set(x for x in a.aug.split(",") if x); CH = a.chans.split(",")
log = open(O / f"log_{a.tag}.txt", "a", encoding="utf-8")
def say(*s):
    m = " ".join(str(x) for x in s); print(m, flush=True); log.write(m + "\n"); log.flush()
say(f"== {a.tag} {vars(a)}")


# ----------------------------------------------------------------------------- data (padded, cached)
def load(split):
    cache = O / f"seq_cache_{split}.npz"
    ids = pd.read_parquet(D / f"clean/features_{split}.parquet", columns=[]).index
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["ids"]) == list(ids): return ids, z["x"], z["n"], z["b"]
    df = pd.read_parquet(D / f"data/{split}_series.parquet")
    df = df.sort_values(["id", "time"], kind="stable"); idv = df["id"].to_numpy()
    u, st, cnt = np.unique(idv, return_index=True, return_counts=True)
    v = df["value"].to_numpy(np.float32); pr = df["period"].to_numpy()
    pos = pd.Series(np.arange(len(u)), index=u).loc[ids].to_numpy()
    x = np.zeros((len(ids), L), np.float32); n = np.zeros(len(ids), np.int64); b = np.zeros(len(ids), np.int64)
    for r, j in enumerate(pos):
        s, c = st[j], cnt[j]; c2 = min(c, L); x[r, :c2] = v[s:s + c2]; n[r] = c2; b[r] = int(pr[s:s + c].argmax())
    np.savez(cache, ids=np.array(list(ids), dtype=object), x=x, n=n, b=b); return ids, x, n, b


ids, X, N, B = load("train"); tids, XT, NT, BT = load("test")
y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label.loc[ids].to_numpy().astype(np.float32)
wt = KU_ENV.diag_weights(O)
say(f"train {X.shape} test {XT.shape} n {N.min()}-{N.max()} b/n {np.median(B / N):.2f}  load {time.time() - t0:.0f}s")
AUXC = [c for c in a.aux.split(",") if c]
if AUXC:   # rank-gauss on train, same map for test (no labels involved)
    at = pd.read_parquet(O / "lm_train.parquet").loc[ids, AUXC].to_numpy(np.float64); ae = pd.read_parquet(O / "lm_test.parquet").loc[tids, AUXC].to_numpy(np.float64)
    def rg(col, ref):
        s = np.sort(ref); q = (np.searchsorted(s, col, side="left") + np.searchsorted(s, col, side="right")) / 2
        return ndtri(np.clip((q + 0.5) / (len(s) + 1), 1e-4, 1 - 1e-4))
    A_tr = np.column_stack([rg(at[:, j], at[:, j]) for j in range(len(AUXC))]).astype(np.float32)
    A_te = np.column_stack([rg(ae[:, j], at[:, j]) for j in range(len(AUXC))]).astype(np.float32)
else: A_tr = np.zeros((len(ids), 0), np.float32); A_te = np.zeros((len(tids), 0), np.float32)
yall = y; esrc = np.array([], dtype=object)
if a.extra:
    assert not AUXC, "--extra needs --aux ''"
    xs, ns, bs_, ys, ss = [X], [N], [B], [y], []
    for st in a.extra.split(","):
        M_ = pd.read_csv(O / "var" / f"{st}_meta.csv", dtype={"id": str, "source": str}); df = pd.read_parquet(O / "var" / f"train_{st}_series.parquet")
        g = {i: (gg["value"].to_numpy(np.float32), gg["period"].to_numpy()) for i, gg in df.groupby("id", sort=False)}
        xe = np.zeros((len(M_), L), np.float32); ne = np.zeros(len(M_), np.int64); be = np.zeros(len(M_), np.int64)
        for r, i in enumerate(M_.id):
            v_, p_ = g[i]; c2 = min(len(v_), L); xe[r, :c2] = v_[:c2]; ne[r] = c2; be[r] = int(p_.argmax())
        xs.append(xe); ns.append(ne); bs_.append(be); ys.append(M_.label.to_numpy(np.float32)); ss.append(M_.source.to_numpy())
    X, N, B, yall = np.concatenate(xs), np.concatenate(ns), np.concatenate(bs_), np.concatenate(ys); esrc = np.concatenate(ss)
    A_tr = np.zeros((len(X), 0), np.float32); say(f"extra rows {len(esrc)} from {a.extra}")
NO = len(ids); id_arr = np.array(list(ids), dtype=object)
gX, gN, gB, gA = [torch.tensor(v, device=dev) for v in (X, N, B, A_tr)]; tX, tN, tB, tA = [torch.tensor(v, device=dev) for v in (XT, NT, BT, A_te)]


# ----------------------------------------------------------------------------- device-side transforms
def view(x, n, b, rev, flip, crop, gen=None):
    """x (Bt,L) padded, n,b (Bt,). Returns cropped / reversed / flipped (x, n, b). crop: bool tensor per row."""
    Bt = x.shape[0]; t = torch.arange(L, device=dev)[None, :]
    if crop is not None and crop.any():
        nb, na = b.float(), (n - b).float(); u1 = torch.rand(Bt, device=dev, generator=gen); u2 = torch.rand(Bt, device=dev, generator=gen)
        s = torch.minimum(u1 * a.crop_max * nb, (nb - 80).clamp(min=0)).floor().long() * crop
        e = torch.minimum(u2 * a.crop_max * na, (na - 80).clamp(min=0)).floor().long() * crop
        idx = (t + s[:, None]).clamp(max=L - 1); x = torch.gather(x, 1, idx); n = n - s - e; b = b - s
    if rev is not None and rev.any():
        idx = (n[:, None] - 1 - t).clamp(min=0); xr = torch.gather(x, 1, idx)
        x = torch.where(rev[:, None], xr, x); b = torch.where(rev, n - b, b)
    x = x * (t < n[:, None])
    if flip is not None: x = torch.where(flip[:, None], -x, x)
    return x, n, b


def channels(x, n, b):
    """(Bt, C, Lb) input channels + masks. Batched AR(0..P) least squares on the before part, BIC order."""
    Bt = x.shape[0]; Lb = int(((int(n.max()) + 63) // 64) * 64); x = x[:, :Lb]; t = torch.arange(Lb, device=dev)[None, :]
    valid = (t < n[:, None]).float(); bef = (t < b[:, None]).float(); aft = valid - bef
    nb = bef.sum(1, keepdim=True); m = (x * bef).sum(1, keepdim=True) / nb; sd = (((x - m) ** 2 * bef).sum(1, keepdim=True) / nb).sqrt().clamp(min=1e-6)
    z = ((x - m) / sd).clamp(-10, 10) * valid
    lags = torch.stack([F.pad(z, (k, 0))[:, :Lb] for k in range(1, P + 1)] + [torch.ones_like(z)], -1)   # (Bt, Lb, P+1)
    rw = (bef * (t >= P).float()); mrow = rw.sum(1)                                                      # rows used in the fit
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
    feats = {"z": z, "e": e, "ae": e.abs(), "e2": (e * e).clamp(max=25) / 5, "ind": aft, "pos": t / n[:, None].float() * valid,
             "dz": torch.cat([torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], 1).clamp(-10, 10) * valid}
    xc = torch.stack([feats[c] for c in CH], 1)
    if a.qin:
        q = torch.cat([mquant(torch.stack([z, e], 1), m[:, None, :], a.qin).flatten(1) for m in (bef, aft)], 1); return xc, bef, aft, q
    return xc, bef, aft, None


def mquant(h, m, nq):
    """masked quantiles over the last axis: h (Bt,C,T), m (Bt,1,T) -> (Bt,C,nq) at levels (i+0.5)/nq."""
    hs = torch.sort(torch.where(m > 0, h, torch.full_like(h, float("inf"))), -1).values; cnt = m.sum(-1, keepdim=True)
    lv = (torch.arange(nq, device=h.device, dtype=h.dtype) + 0.5) / nq; ix = (lv[None, None, :] * (cnt - 1)).round().long()
    return torch.gather(hs, -1, ix.expand(-1, h.shape[1], -1))


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(s, c, d, k, p):
        super().__init__(); s.n1 = nn.BatchNorm1d(c); s.c1 = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d)
        s.n2 = nn.BatchNorm1d(c); s.c2 = nn.Conv1d(c, c, k, padding=d * (k // 2), dilation=d); s.dp = nn.Dropout(p)
    def forward(s, h, m): return (h + s.c2(s.dp(F.gelu(s.n2(s.c1(F.gelu(s.n1(h)))))))) * m


class Net(nn.Module):
    def __init__(s, cin, c, dil, k, drop, naux):
        super().__init__(); nq = 2 * 2 * a.qin; s.qn = nn.Linear(3 * nq // 2, 64) if a.qin else None
        npool = (2 + (a.hq if a.hq else 1)) * c; s.stem = nn.Conv1d(cin, c, 7, padding=3); s.blocks = nn.ModuleList([Block(c, d, k, 0.1) for d in dil]); s.nf = nn.BatchNorm1d(c)
        s.head = nn.Sequential(nn.Linear(4 * npool + 2 + naux + (64 if a.qin else 0), 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop / 2), nn.Linear(64, 1))
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
        extra = []
        if q is not None:
            qb, qa = q.float().chunk(2, 1); extra = [F.gelu(s.qn(torch.cat([qb, qa, qa - qb], 1)))]
        return s.head(torch.cat([pb, pa, pa - pb, (pa - pb).abs(), ln, aux] + extra, 1))[:, 0]


def run(model, x, n, b, aux, rev=None, flip=None, crop=None, gen=None):
    xv, nv, bv = view(x, n, b, rev, flip, crop, gen); xc, bef, aft, q = channels(xv, nv, bv)
    return model(xc, bef, aft, aux, q).float()          # Full float32 on CPU; --bf16 is accepted and ignored.


@torch.no_grad()
def predict(model, x, n, b, aux, bs=512):
    model.eval(); out = []; srt = torch.argsort(n); inv = torch.argsort(srt).cpu().numpy(); x, n, b, aux = x[srt], n[srt], b[srt], aux[srt]
    for i in range(0, len(n), bs):
        sl = slice(i, i + bs); r = []
        for rv in (False, True):
            for fl in (False, True):
                if (rv and "rev" not in AUG) or (fl and "flip" not in AUG): continue
                k = n[sl].shape[0]; r.append(run(model, x[sl], n[sl], b[sl], aux[sl], torch.full((k,), rv, device=dev), torch.full((k,), fl, device=dev)))
        out.append(torch.stack(r, 0))
    return torch.cat(out, 1).cpu().numpy()[:, inv]   # (views, rows) logits, original order


def train_fold(tr, seed):
    torch.manual_seed(seed); np.random.seed(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    tri, ini = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=seed)
    if len(esrc): tri = np.concatenate([tri, NO + np.where(np.isin(esrc, id_arr[tri]))[0]])
    model = Net(len(CH), a.ch, [int(d) for d in a.dil.split(",")], a.k, a.drop, len(AUXC)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); steps = a.epochs * math.ceil(len(tri) / a.bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    yt = torch.tensor(yall, device=dev); best, best_ep, best_state, bad = -1, -1, None, 0; tri_t = torch.tensor(tri, device=dev); ini_t = torch.tensor(ini, device=dev)
    for ep in range(a.epochs):
        model.train(); perm = tri_t[torch.randperm(len(tri), device=dev, generator=gen)]; tl = 0.0
        chunks = [c[torch.argsort(gN[c])] for c in perm.split(a.bs * 16)]          # length buckets: less padding
        batches = [bb for c in chunks for bb in c.split(a.bs)]; order = torch.randperm(len(batches), generator=torch.Generator().manual_seed(seed * 100 + ep))
        for bi in order.tolist():
            j = batches[bi]; k = len(j)
            if k < 8: continue
            rv = (torch.rand(k, device=dev, generator=gen) < 0.5) if "rev" in AUG else None
            fl = (torch.rand(k, device=dev, generator=gen) < 0.5) if "flip" in AUG else None
            cr = (torch.rand(k, device=dev, generator=gen) < a.crop_p) if "crop" in AUG else None
            lo = run(model, gX[j], gN[j], gB[j], gA[j], rv, fl, cr, gen); loss = F.binary_cross_entropy_with_logits(lo, yt[j])
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sch.step(); tl += loss.item() * k
        pin = predict(model, gX[ini_t], gN[ini_t], gB[ini_t], gA[ini_t]).mean(0); auc = roc_auc_score(y[ini], pin)
        if auc > best: best, best_ep, bad = auc, ep, 0; best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else: bad += 1
        say(f"   ep {ep:2d} loss {tl / len(tri):.4f} inner {auc:.4f}{' *' if bad == 0 else ''}  {time.time() - t0:.0f}s")
        if bad >= a.patience: break
    model.load_state_dict(best_state); return model, best, best_ep


PD = O / "seqpart"
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)); run_f = a.folds if a.folds else range(5)
oof = np.full(len(y), np.nan); oofv = np.full((4, len(y)), np.nan); tst = []; meta = []
for f in run_f:
    tr, va = folds[f]; vl, tl_ = [], []
    for sd in a.seeds:
        model, inner, ep = train_fold(tr, 1000 * f + sd); va_t = torch.tensor(va, device=dev)
        WD = D / 'weights' / a.tag
        if __import__('os').environ.get('SAVE_WEIGHTS', '1') != '0': WD.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), WD / f'fold{f}_seed{sd}.pt')
        pv = predict(model, gX[va_t], gN[va_t], gB[va_t], gA[va_t]); vl.append(pv); tl_.append(predict(model, tX, tN, tB, tA))
        say(f"fold {f} seed {sd}: inner {inner:.4f} @ep {ep}  val plain {roc_auc_score(y[va], pv[0]):.5f}  tta {roc_auc_score(y[va], pv.mean(0)):.5f}")
        meta.append(dict(fold=f, seed=sd, inner=inner, ep=ep))
    pv = np.mean(vl, 0); oof[va] = pv.mean(0); oofv[:pv.shape[0], va] = pv; tst.append(np.mean(tl_, 0).mean(0))
    say(f"fold {f}: val tta {roc_auc_score(y[va], oof[va]):.5f}  {time.time() - t0:.0f}s")
    PD.mkdir(exist_ok=True); np.savez(PD / f"{a.tag}_f{f}.npz", va=va, pv=pv, pt=np.mean(tl_, 0), meta=json.dumps([m for m in meta if m["fold"] == f]))
# assemble from per-fold parts (folds may have been run by separate processes)
if all((PD / f"{a.tag}_f{f}.npz").exists() for f in range(5)) and len(run_f) < 5:
    tst, meta = [], []
    for f in range(5):
        z = np.load(PD / f"{a.tag}_f{f}.npz"); oof[z["va"]] = z["pv"].mean(0); oofv[:z["pv"].shape[0], z["va"]] = z["pv"]; tst.append(z["pt"].mean(0)); meta += json.loads(str(z["meta"]))
    run_f = range(5); say("assembled 5 folds from parts")
sig = lambda z: 1 / (1 + np.exp(-z))
if len(run_f) == 5:
    cv, cvw = roc_auc_score(y, oof), roc_auc_score(y, oof, sample_weight=wt); cv0 = roc_auc_score(y, oofv[0])
    say(f"{a.tag}: CV tta {cv:.5f} / {cvw:.5f} (test-weighted); plain {cv0:.5f}  {time.time() - t0:.0f}s")
    np.save(O / f"oof_{a.tag}.npy", sig(oof)); np.save(O / f"test_{a.tag}.npy", sig(np.mean(tst, 0))); np.save(O / f"oofviews_{a.tag}.npy", oofv)
    json.dump(dict(tag=a.tag, cv=cv, cv_test_weighted=cvw, cv_plain=cv0, args=vars(a), folds=meta, secs=time.time() - t0), open(O / f"best_{a.tag}.json", "w"), indent=1)
else:
    m = ~np.isnan(oof); say(f"{a.tag} partial folds {list(run_f)}: tta {roc_auc_score(y[m], oof[m]):.5f}")
