"""try3 fast path for white_model_features.py hetero/memory1, late_features.py native_law_*/latent_level_* and
mechanism.py volatility_law.

each slim_* function is a copy of the original extract() (same tensors, same order, same dtypes, same batching) that
skips only work whose results never reach the columns the models in use read (slim_cols.needed, KU_TRY3_MODEL), so
every kept value is identical. supported() is False when a column cannot be named here (the original step runs then).

python try3/slim_white.py hetero|memory1 test              (white_model_features.py kinds)
python try3/slim_white.py native_law_<stream>|latent_level_<stream> test   (late_features.py kinds)
python try3/slim_white.py volatility_law --part test
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slim_cols


def frame(rows, ids, cols, B_list):
    #stacked float32 rows -> DataFrame; a table with no kept column still gets its index
    return pd.DataFrame(np.vstack(rows) if cols else np.zeros((len(ids), 0), np.float32), index=ids, columns=cols)


# ---------------------------------------------------------------------------------------------- hetero_jump
def hetero_plan(need):
    plan = {}
    for c in need:
        ctx, _, rest = c.partition("_")
        if ctx not in ("split", "common") or not re.fullmatch(r"(level|scale)\d(_law\d)?|shape|pre\d", rest):
            raise KeyError(c)
        plan.setdefault(ctx, []).append(rest)
    return plan


def slim_hetero(m, part):
    #white_hetero_shift: profiles() unchanged (hetero_path is also read by filtered_step); extract() only the kernels,
    #contexts and summaries the kept columns use
    import torch
    need = slim_cols.needed("hetero_jump")
    plan = hetero_plan(need)
    m.profiles(part)
    dst = m.OUT / f"hetero_jump_{part}.parquet"
    with np.load(m.OUT / f"hetero_path_{part}.npz") as a:
        ids, u, n, b, h, h0 = (a[k] for k in ("ids", "u", "n", "b", "separate", "common"))
    bank = m.dictionary()
    mass = m.tensor(m.MASS[m.KSEL] / m.MASS[m.KSEL].sum()).log()
    lse = torch.logsumexp
    MU, S, KSEL = m.MU, m.S, m.KSEL
    wanted = {r.split("_")[0] for rs in plan.values() for r in rs if r.startswith(("level", "scale"))}
    loc = {j: m.jump_kernel(MU, np.exp(S), d, w) for j, (d, w) in enumerate(((0.0, 0.08), (0.0, 0.22), (0.0, 0.52), (0.0, 1.15), (0.35, 0.15), (1.0, 0.34))) if f"level{j}" in wanted}
    vel = {j: m.jump_kernel(S, np.ones(1), d, w) for j, (d, w) in enumerate(((0.0, 0.07), (0.0, 0.2), (0.0, 0.48), (0.32, 0.12), (0.6, 0.17))) if f"scale{j}" in wanted}
    rows = []
    for start in range(0, len(n), 48):
        sl = slice(start, min(len(n), start + 48))
        feats = {}
        for hfield, context in ((h, "split"), (h0, "common")):
            if context not in plan:
                continue
            rest = plan[context]
            counts, md, s = map(np.asarray, zip(*(m.chart(z[:end].astype(float), v[:end], j) for z, v, end, j in zip(u[sl], hfield[sl], n[sl], b[sl]))))
            B = len(md)
            ll = (m.tensor(counts).reshape(-1, bank.shape[0]) @ bank).reshape(B, 2, len(KSEL), len(MU), len(S))
            ll -= ll.amax((2, 3, 4), keepdim=True)
            a, c = ll.unbind(1)
            absolute = md[:, None] + s[:, None] * MU
            logs = np.log(s[:, None]) + S
            mu = m.tensor(np.log(0.89 / 4 / (1 + np.exp(((abs(absolute) - 2) / 0.075).clip(-35, 35))) + 0.11 * np.exp(-abs(absolute) / 2) / 4) + np.log(s[:, None] * np.diff(MU)[0]))
            sd = m.tensor(np.log(0.88 / 1.4 / (1 + np.exp(((abs(logs) - 0.7) / 0.045).clip(-35, 35))) + 0.12 * np.exp(-abs(logs)) * 0.5) + np.log(np.diff(S)[0]))
            pm = mu[:, None, :, None]
            ps = sd[:, None, None, :]
            null = lse(a + c + pm + ps, (2, 3))
            z0 = lse(null + mass, 1)
            if any("law" in r or r.startswith("pre") for r in rest):
                pre = (lse(a + pm + ps, (2, 3)) + mass).softmax(1)
            for prefix, ks in (("level", loc), ("scale", vel)):
                for j, ker in ks.items():
                    if not any(r.split("_")[0] == f"{prefix}{j}" for r in rest):
                        continue
                    if prefix == "level":
                        sc = m.integrated_shift(a.transpose(-1, -2), c.transpose(-1, -2), mu[:, None, None, :], sd[:, None, :], ker)
                    else:
                        sc = m.integrated_shift(a, c, ps, mu[:, None, :], ker)
                    feats[f"{context}_{prefix}{j}"] = lse(sc + mass, 1) - z0
                    if j in (1, 3):
                        for k in range(len(KSEL)):
                            if f"{prefix}{j}_law{k}" in rest:
                                feats[f"{context}_{prefix}{j}_law{k}"] = pre[:, k] * (sc[:, k] - null[:, k]).clip(-25, 70)
            if "shape" in rest:
                af, cf = [lse(v + mass[None, :, None, None], 1) for v in (a, c)]
                feats[context + "_shape"] = lse(af + cf + mu[:, :, None] + sd[:, None, :], (1, 2)) - z0
            for k in range(len(KSEL)):
                if f"pre{k}" in rest:
                    feats[f"{context}_pre{k}"] = pre[:, k]
        if need:
            rows.append(torch.stack([feats[c] for c in need], 1).clip(-100, 160).cpu().numpy())
    out = frame(rows, ids, need, None)
    out.index.name = "id"
    assert np.isfinite(out).all().all()
    out.to_parquet(dst)
    print(part, "heteroscedastic physical shifts (slim)", out.shape, flush=True)


# ---------------------------------------------------------------------------------------------- memory_law<lag>
MIXED = ("independent_memory", "jump1", "jump2", "zero_start", "zero_end")


def memory_plan(need, lag):
    want = set(need)
    if "mixed" in want:
        want |= set(MIXED)
    for c in want:
        ok = c in ("independent_memory", "fixed_memory_new_scale", "pre_pacf", "mixed") or re.fullmatch(r"jump[0-4]", c) \
            or re.fullmatch(r"law\d_jump[023]", c) or re.fullmatch(r"(zero|unit)_(both|start|end)", c)
        if not ok or (lag == 2 and c.startswith("unit_")):
            raise KeyError(c)
    return want


def slim_memory(m, part, lag, bankmat):
    #white_memory_law.extract(): only the kernels / fixed-memory variants / law splits the kept columns use
    #(mixed still combines independent_memory, jump1, jump2, zero_start, zero_end in the original order)
    import torch
    from scipy.special import expit
    need = slim_cols.needed(f"memory_law{lag}")
    want = memory_plan(need, lag)
    path = m.OUT / f"memory_law{lag}_{part}.parquet"
    with np.load(m.OUT / f"pulse_streams_{part}.npz") as q:
        ids, x, n, b = (q[k] for k in ("ids", "separate", "n", "b"))
    phi = pd.read_parquet(m.OUT / f"state_{part}.parquet").loc[ids, ["ar1", "ar2"]].to_numpy()
    D, S, IDS = m.D, m.S, m.IDS
    pm = m.tensor(np.log(m.P))
    lse = torch.logsumexp
    delta = D[None, :] - D[:, None]
    wd = np.gradient(D)
    sizes = [(0.0, 0.06), (0.0, 0.16), (0.0, 0.36), (0.25, 0.09), (0.5, 0.15)]
    kernels = [(np.exp(-0.5 * ((delta - r) / s) ** 2) + np.exp(-0.5 * ((delta + r) / s) ** 2)) * wd[None, :] for r, s in sizes]
    jumps = [j for j in range(5) if f"jump{j}" in want or any(c.endswith(f"_jump{j}") and c.startswith("law") for c in want)]
    rows = []
    for start in range(0, len(x), 60):
        h, sc, ref = [], [], []
        for r, j, e, p in zip(x[start:start + 60], b[start:start + 60], n[start:start + 60], phi[start:start + 60]):
            hh, ss, rr = m.row(r[:e].astype(float), j, p, lag)
            h.append(hh)
            sc.append(ss)
            ref.append(rr)
        B = len(h)
        val = (m.tensor(np.asarray(h)).reshape(B * 2, -1) @ bankmat).reshape(B, 2, len(IDS), len(D), len(S))
        val -= val.amax((2, 3, 4), keepdim=True)
        a, c = val.unbind(1)
        ref = np.array(ref)
        sc = np.array(sc)
        bounded = expit((0.96 - abs(D[None, :] + ref[:, None])) / 0.025)
        uni = bounded * wd[None, :]
        uni /= uni.sum(1, keepdims=True)
        spike = m.point(-ref)
        unit = m.point(1 - ref)
        prior = 0.54 * spike + 0.38 * uni + 0.08 * unit if lag == 1 else 0.9 * spike + 0.1 * uni
        pdyn = m.tensor(np.log(prior))[:, None, :, None]
        logsd = np.log(sc[:, None]) + S
        sv = 0.9 * expit((0.71 - abs(logsd)) / 0.05) + 0.1 * np.exp(-abs(logsd))
        ps = m.tensor(np.log(sv / sv.sum(1, keepdims=True)))[:, None, :]
        null = lse(lse(a + c + pdyn, 2) + ps, 2)
        z0 = lse(null + pm, 1)
        aw = lse(a + pdyn, 2)
        bw = lse(c + pdyn, 2)

        def total(v):
            return lse(v + pm, 1) - z0
        f = {}
        if "independent_memory" in want:
            free = lse(aw + bw + ps, 2)
            f["independent_memory"] = total(free)
        if "fixed_memory_new_scale" in want:
            sep = lse(lse(a + ps[:, :, None, :], 3) + lse(c + ps[:, :, None, :], 3) + pdyn[..., 0], 2)
            f["fixed_memory_new_scale"] = total(sep)
        if "pre_pacf" in want:
            f["pre_pacf"] = m.tensor(ref)
        if jumps:
            left = a + pdyn
            ma, mb = (left.amax(2, keepdim=True), c.amax(2, keepdim=True))
            ea = (left - ma).exp().transpose(2, 3).contiguous().reshape(B, -1, len(D))
            eb = (c - mb).exp().transpose(2, 3)
        for j in jumps:
            K = kernels[j][None, :, :] * bounded[:, None, :]
            K = K / K.sum(2, keepdims=True).clip(1e-12)
            acc = (ea @ m.tensor(K)).reshape(B, len(IDS), len(S), len(D))
            ss = (acc * eb).sum(3).clamp_min(1e-37).log() + ma[:, :, 0, :] + mb[:, :, 0, :]
            v = lse(ss + ps, 2)
            f[f"jump{j}"] = total(v)
            if j in (0, 2, 3) and any(cc.startswith("law") and cc.endswith(f"_jump{j}") for cc in want):
                prelaw = (lse(aw + ps, 2) + pm).softmax(1)
                for i in range(len(IDS)):
                    if f"law{i}_jump{j}" in want:
                        f[f"law{i}_jump{j}"] = prelaw[:, i] * (v[:, i] - null[:, i]).clip(-18, 40)
        for nm, fixed in (("zero", spike), ("unit", unit)):
            if lag == 2 and nm == "unit":
                continue
            ks = [k for k in ("both", "start", "end") if f"{nm}_{k}" in want]
            if not ks:
                continue
            lp = m.tensor(np.log(fixed))[:, None, :, None]
            pre, post = (lse(a + lp, 2), lse(c + lp, 2))
            for k, v in (("both", lambda: pre + post), ("start", lambda: pre + bw), ("end", lambda: aw + post)):
                if k in ks:
                    f[nm + "_" + k] = total(lse(v() + ps, 2))
        if "mixed" in want:
            f["mixed"] = lse(torch.stack([f[k] for k in MIXED], 1), 1) - np.log(5)
        if need:
            rows.append(torch.stack([f[k] for k in need], 1).clip(-80, 100).cpu().numpy())
    df = frame(rows, pd.Index(ids, name="id"), need, None)
    assert np.isfinite(df).all().all()
    df.to_parquet(path)
    print("PACF", lag, part, df.shape, "(slim)", flush=True)


# ---------------------------------------------------------------------------------------------- native_law_<stream>
def native_plan(need):
    #per bank (True = native_*, False = standard_*) the kept record names
    out = {True: [], False: []}
    for c in need:
        pre, _, rest = c.partition("_")
        if pre not in ("native", "standard") or not re.fullmatch(r"(fixed|free_scale|free_mean)(_(from|to)\d)?|pair\d_\d|(pre|post|null)\d", rest):
            raise KeyError(c)
        out[pre == "native"].append(rest)
    return out


def slim_native(m, part, stream):
    #white_native_law.extract(): per bank only the transition types / summaries the kept columns use
    import torch
    need_all = slim_cols.needed(f"native_law_{stream}")
    plan = native_plan(need_all)
    dst = m.OUT / f"native_law_{stream}_{part}.parquet"
    with np.load(m.OUT / f"{stream}_streams_{part}.npz") as f:
        ids, u, n, b = (f[k] for k in ("ids", "separate", "n", "b"))
    counts, mm, ss = [], [], []
    for row, end, j in zip(u, n, b):
        x = row[:end].astype(float)
        center = np.median(x)
        sd = max(np.median(abs(x - center)) * 1.4826, 0.18 * x.std(), 0.01)
        z = (x - center) / sd
        counts.append([m.hist(z[:j]), m.hist(z[j:])])
        mm.append(center)
        ss.append(sd)
    counts = np.asarray(counts, np.float32)
    mm, ss = (np.array(mm), np.array(ss))
    M, S, FAM, MASS, AXIS = m.M, m.S, m.FAM, m.MASS, m.AXIS
    blocks = []
    for native in (True, False):
        need = plan[native]
        if not need:
            continue
        types = [t for t in ("fixed", "free_scale", "free_mean")
                 if any(c == t or c.startswith(t + "_") for c in need) or (t == "fixed" and any(c.startswith("pair") for c in need))]
        #fixed joint: when only pair<i>_* of one row i are read (and not fixed/fixed_from/fixed_to), only row i is built
        pairs = {int(c[4]) for c in need if c.startswith("pair")}
        fixed_row = pairs.pop() if len(pairs) == 1 and not any(c == "fixed" or c.startswith("fixed_") for c in need) else None
        sides = [sd_ for sd_ in ("pre", "post") if any(re.fullmatch(sd_ + r"\d", c) for c in need)]
        nulls = [int(c[4:]) for c in need if re.fullmatch(r"null\d", c)]
        use_a = bool(types) or "pre" in sides
        B, groups = m.bank(native)
        idx = [np.flatnonzero(groups == k) for k in range(len(FAM))]
        pi = m.tensor([MASS[k] / len(idx[k]) for k in groups]).log()
        pf = m.tensor(MASS).log()
        rows = []
        lse = torch.logsumexp
        for start in range(0, len(mm), 32):
            sl = slice(start, min(start + 32, len(mm)))
            N = len(mm[sl])
            ll = (m.tensor(counts[sl]).reshape(-1, len(AXIS)) @ B).reshape(N, 2, len(groups), len(M), len(S))
            ll -= ll.amax((2, 3, 4), keepdim=True)
            a, c = ll.unbind(1)
            mu = mm[sl, None] + ss[sl, None] * M
            logs = np.log(ss[sl, None]) + S
            pm = m.tensor(np.log(0.86 / 4 / (1 + np.exp(((abs(mu) - 2) / 0.085).clip(-35, 35))) + 0.14 * np.exp(-abs(mu) / 2) / 4) + np.log(ss[sl, None] * np.diff(M)[0]))
            lo = -1.2 if native else -0.7
            prior = 0.82 / (0.7 - lo) / (1 + np.exp(((lo - logs) / 0.08).clip(-35, 35))) / (1 + np.exp(((logs - 0.7) / 0.08).clip(-35, 35))) + 0.18 * np.exp(-abs(logs)) * 0.5
            ps = m.tensor(np.log(prior) + np.log(np.diff(S)[0]))
            pm0 = pm[:, None, :, None]
            ps0 = ps[:, None, None, :]
            records = {}
            if types or nulls:
                null = lse(a + c + pm0 + ps0, (2, 3))
                total = lse(null + pi, 1)
            C = torch.stack([lse(c[:, j], 1) - np.log(len(j)) for j in idx], 1)
            A = torch.stack([lse(a[:, j], 1) - np.log(len(j)) for j in idx], 1) if use_a else None

            def collapse(t, p, ax):
                return lse(t + p, ax)
            for typ in types:
                result = []
                #collapse(C, ...) does not depend on i: computed once (same op on the same tensor -> same values)
                cs = collapse(C, ps0, 3) if typ == "free_scale" else collapse(C, pm0, 2) if typ == "free_mean" else None
                only = fixed_row if typ == "fixed" else None
                for i in (range(len(FAM)) if only is None else [only]):
                    j = A[:, i:i + 1]
                    if typ == "fixed":
                        result.append(lse(j + C + pm0 + ps0, (2, 3)))
                    elif typ == "free_scale":
                        result.append(lse(collapse(j, ps0, 3) + cs + pm[:, None, :], 2))
                    else:
                        result.append(lse(collapse(j, pm0, 2) + cs + ps[:, None, :], 2))
                if only is None:
                    joint = torch.stack(result, 1) + pf[None, :, None] + pf[None, None, :]
                    records[typ] = lse(joint, (1, 2)) - total
                    for i in range(len(FAM)):
                        if f"{typ}_from{i}" in need:
                            records[f"{typ}_from{i}"] = lse(joint[:, i], 1) - total
                        if f"{typ}_to{i}" in need:
                            records[f"{typ}_to{i}"] = lse(joint[:, :, i], 1) - total
                        if typ == "fixed":
                            for k in range(len(FAM)):
                                records[f"pair{i}_{k}"] = joint[:, i, k] - total
                else:
                    joint = torch.stack(result, 1) + pf[None, only:only + 1, None] + pf[None, None, :]
                    for k in range(len(FAM)):
                        records[f"pair{only}_{k}"] = joint[:, 0, k] - total
            for side, v in (("pre", A), ("post", C)):
                if side in sides:
                    probs = (lse(v + pm0 + ps0, (2, 3)) + pf).softmax(1)
                    for j in range(len(FAM)):
                        records[f"{side}{j}"] = probs[:, j]
            for k in nulls:
                ii = idx[k]
                records[f"null{k}"] = lse(null[:, ii] + pi[ii], 1) - total
            arr = torch.stack([records[c] for c in need], 1).clip(-55, 110).cpu().numpy()
            rows.append(pd.DataFrame(arr, index=ids[sl], columns=need))
        blocks.append(pd.concat(rows).add_prefix("native_" if native else "standard_"))
        del B
    out = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=ids)
    out.index.name = "id"
    assert np.isfinite(out).all().all()
    out.to_parquet(dst)
    print(part, stream, "asymmetry and physical scale (slim)", out.shape, flush=True)


# ---------------------------------------------------------------------------------------------- latent_level_<stream>
ALT = ("observed_step", "kick", "driver", "slope", "pulse")
MASKS = ("no_memory", "walk", "hidden", "slow_hidden", "negative")


def latent_plan(need):
    kinds = set()
    restart = False
    for c in need:
        if mt := re.fullmatch(r"(observed_step|kick|driver|slope|pulse)([0-2])", c):
            kinds.add(ALT.index(mt.group(1)))
        elif c in ("restart_state", "restart_new_scale", "restart_new_process"):
            restart = True
        elif mt := re.fullmatch(r"(no_memory|walk|hidden|slow_hidden|negative)_(full|pre|post)", c):
            restart |= mt.group(2) != "full"
        else:
            raise KeyError(c)
    return tuple(sorted(kinds)), restart


def slim_fit(m, x, n, b, bank, intervene, kinds=()):
    #white_latent_level.fit() with only the alternative designs in `kinds` (each alternative is computed independently)
    import torch
    H, u, step, accum, extra, _ = bank
    gaussian = m.gaussian
    B, T = x.shape
    t = torch.arange(T, device=x.device)[None, :]
    present = (t < n[:, None])[:, None, :]
    y = torch.fft.irfft(torch.fft.rfft(x, n=2 * T)[:, None, :] * H[None, :, :], n=2 * T)[:, :, :T]
    U = u[None, :, :] * present
    factor = extra[None, :] / (1 + extra[None, :] * (U * U).sum(2))
    base_det = torch.log1p(extra[None, :] * (U * U).sum(2))

    def prod(x, z):
        return (x * z * present).sum(2) - factor * (x * U).sum(2) * (z * U).sum(2)
    ds = [step[None, :, :], step.cumsum(1)[None, :, :] / n[:, None, None]]
    alt = {}
    if intervene:
        j = t - b[:, None]
        active = (j >= 0)[:, None, :]
        ii = j.clamp_min(0)

        def move(v):
            return v[:, ii].permute(1, 0, 2) * active

        def pulse():
            later = (j - 6).clamp_min(0)
            delayed = step[:, later].permute(1, 0, 2) * (j >= 6)[:, None, :]
            return move(step) - delayed
        make = {0: lambda: move(step), 1: lambda: move(u), 2: lambda: move(accum), 3: lambda: move(step.cumsum(1)) / n[:, None, None], 4: pulse}
        alt = {k: make[k] for k in kinds}
    yy = prod(y, y)
    z = prod(ds[0], y)
    g = prod(ds[0], ds[0]) + 1 / 2.5 ** 2
    sse = yy - z * z / g
    det = base_det + torch.log(g * 2.5 ** 2)
    tdy = prod(ds[1], y)
    cross = prod(ds[1], ds[0])
    tdy = tdy - cross * z / g
    tg = (prod(ds[1], ds[1]) - cross.square() / g).clamp_min(0) + 1 / 1.5 ** 2
    ss1 = (sse - tdy.square() / tg).clamp_min(0)
    dt1 = det + torch.log(tg * 1.5 ** 2)
    alts = {}
    for k, mk in alt.items():
        d = mk()
        r0 = prod(d, ds[0])
        xy = prod(d, y) - r0 * z / g
        xx = (prod(d, d) - r0.square() / g).clamp_min(0)
        rt = prod(d, ds[1]) - r0 * cross / g
        for label, s, ld, xy1, xx1 in ((0, sse, det, xy, xx), (1, ss1, dt1, xy - rt * tdy / tg, (xx - rt.square() / tg).clamp_min(0))):
            items = []
            for sd in (0.05, 0.18, 0.55) if k == 2 else (0.25, 0.75, 1.65):
                I = xx1 + sd ** (-2)
                items.append(gaussian(s - xy1.square() / I, ld + torch.log(I * sd ** 2), n[:, None]))
            alts[k, label] = torch.stack(items, 2)
        del d
    return (sse.clamp_min(0), det, ss1, dt1, alts)


def slim_calc(m, x, n, b, bank, need, kinds, restart):
    import torch
    H, u, step, accum, extra, logp = bank
    gaussian, blend = m.gaussian, m.blend
    B, T = x.shape
    s, d, st, dt, alts = slim_fit(m, x, n, b, bank, True, kinds)
    base = blend(gaussian(s, d, n[:, None]), gaussian(st, dt, n[:, None]))
    lp = base + logp
    null = torch.logsumexp(lp, 1)
    post = lp.softmax(1)
    out = {}
    for k in kinds:
        tag = ALT[k]
        L = blend(alts[k, 0], alts[k, 1]) + logp[None, :, None]
        odds = L.logsumexp(1) - null[:, None]
        for i in range(3):
            out[tag + str(i)] = odds[:, i]
    if restart:
        pos = torch.arange(T, device=x.device)[None, :] + b[:, None]
        other = x.gather(1, pos.clamp_max(T - 1)).masked_fill(pos >= n[:, None], 0)
        a, ad, _, _, _ = slim_fit(m, x, b, b, bank, False)
        c, cd, _, _, _ = slim_fit(m, other, n - b, b, bank, False)
        pre = gaussian(a, ad, b[:, None])
        after = gaussian(c, cd, (n - b)[:, None])
        shared = gaussian(a + c, ad + cd, n[:, None])
        out["restart_state"] = (shared + logp).logsumexp(1) - null
        out["restart_new_scale"] = (pre + after + logp).logsumexp(1) - null
        out["restart_new_process"] = (pre + logp).logsumexp(1) + (after + logp).logsumexp(1) - null
    P = np.array(m.MODELS)
    for tag, mask in (("no_memory", P[:, 0] == 0), ("walk", P[:, 0] == 1), ("hidden", P[:, 1] < 0.51), ("slow_hidden", (P[:, 1] < 0.51) & (P[:, 0] > 0.9)), ("negative", P[:, 0] < 0)):
        for nm in ("full", "pre", "post"):
            if f"{tag}_{nm}" in need:
                p = post if nm == "full" else (pre + logp).softmax(1) if nm == "pre" else (after + logp).softmax(1)
                out[tag + "_" + nm] = p[:, mask].sum(1)
    return out


def slim_latent(m, stream, part):
    #white_latent_level.extract()
    import torch
    need = slim_cols.needed(f"latent_level_{stream}")
    kinds, restart = latent_plan(need)
    dst = m.OUT / f"latent_level_{stream}_{part}.parquet"
    path = m.ROOT / f"tune_out/seq_cache_{part}.npz" if stream == "raw" else m.OUT / f"{stream}_streams_{part}.npz"
    with np.load(path, allow_pickle=True) as q:
        ids, x, n, b = [q[k] for k in ("ids", "x" if stream == "raw" else "separate", "n", "b")]
    m.validate(x[0])
    T = x.shape[1]
    bank = m.kernels(T)
    rows = []
    for j in range(0, len(x), 70):
        sl = slice(j, j + 70)
        nn = torch.tensor(n[sl], device="cpu")
        bb = torch.tensor(b[sl], device="cpu")
        rec = slim_calc(m, m.tensor(x[sl]), nn, bb, bank, need, kinds, restart)
        arr = torch.stack([rec[c] for c in need], 1).clamp(-80, 120).cpu().numpy() if need else np.zeros((len(ids[sl]), 0), np.float32)
        rows.append(pd.DataFrame(arr, index=ids[sl].astype(str), columns=need))
    result = pd.concat(rows)
    result.index.name = "id"
    assert np.isfinite(result).all().all()
    result.to_parquet(dst)
    print(stream, part, "latent-level evidence (slim)", result.shape, flush=True)


# ---------------------------------------------------------------------------------------------- volatility_law
VOL_FIXED = ("pre_cluster", "post_cluster", "pooled_cluster", "all", "new_memory", "new_law", "new_scale", "memory_on",
             "memory_off", "memory_and_scale", "memory_and_law", "scale_and_law")


def vol_check(need):
    for c in need:
        if not (c in VOL_FIXED or re.fullmatch(r"scale_(jump|cluster)[0-4]|(pre|post)_decay[0-4]|law[0-8]_(null|memory)", c)):
            raise KeyError(c)


def slim_volatility(m, part):
    #white_volatility_law.extract(): every column by name with the original expressions; shared pieces built once,
    #only when a kept column uses them
    import torch
    need = slim_cols.needed("volatility_law")
    vol_check(need)
    E, H, LOGS, DECAY, MIX, K = m.E, m.H, m.LOGS, m.DECAY, m.MIX, m.K
    path = m.OUT / f"volatility_law_{part}.parquet"
    with np.load(m.OUT / f"pulse_streams_{part}.npz") as q:
        ids, x, n, b = (q[k] for k in ("ids", "separate", "n", "b"))
    phi = pd.read_parquet(m.OUT / f"state_{part}.parquet").loc[ids, ["ar1", "ar2"]].to_numpy()
    bank = m.table()
    pdyn = m.tensor(m.PRIOR).log()
    pl = m.tensor(m.MASS[K] / m.MASS[K].sum()).log()
    S = len(LOGS)
    lse = torch.logsumexp
    jk = {int(c[-1]) for c in need if re.fullmatch(r"scale_(jump|cluster)\d", c)}
    kernels = {}
    for j, (u, w) in enumerate(((0.0, 0.07), (0.0, 0.2), (0.0, 0.48), (0.32, 0.12), (0.6, 0.17))):
        if j not in jk:
            continue
        delta = LOGS[:, None] - LOGS[None, :]
        ker = (np.exp(-0.5 * ((delta - u) / w) ** 2) + np.exp(-0.5 * ((delta + u) / w) ** 2)) * np.diff(LOGS)[0] / (2 * np.sqrt(2 * np.pi) * w)
        kernels[j] = m.tensor(ker).clamp_min(1e-30)
    rows = []
    for start in range(0, len(ids), 70):
        sl = slice(start, min(start + 70, len(ids)))
        obs, scale = map(np.asarray, zip(*(m.prepare(row[:end].astype(float), j, ar) for row, end, j, ar in zip(x[sl], n[sl], b[sl], phi[sl]))))
        B = len(scale)
        ll = (m.tensor(obs).reshape(-1, len(E) * len(H)) @ bank).reshape(B, len(DECAY), 2, len(MIX), len(K), S).transpose(1, 2)
        ll = torch.cat([ll[:, :, 0, 0:1], ll[:, :, :, 1:].flatten(2, 3)], 2)
        ll -= ll.amax((2, 3, 4), keepdim=True)
        a, c = ll.unbind(1)
        logs = np.log(scale[:, None]) + LOGS
        ps = m.tensor(np.log(0.88 / 1.4 / (1 + np.exp(((abs(logs) - 0.7) / 0.06).clip(-30, 30))) + 0.12 * np.exp(-abs(logs)) * 0.5) + np.log(np.diff(LOGS)[0]))
        prior = pdyn[None, :, None, None] + pl[None, None, :, None] + ps[:, None, None, :]
        joint = a + c + prior
        z0 = lse(joint, (1, 2, 3))
        dctx = pdyn[None, :, None] + pl[None, None, :]
        sctx = pl[None, :, None] + ps[:, None, :]
        dscale = pdyn[None, :, None] + ps[:, None, :]
        cache = {}

        def get(key, fn):
            if key not in cache:
                cache[key] = fn()
            return cache[key]

        def prob(v):
            return lse(v + prior, (2, 3)).softmax(1)
        pre = lambda: get("pre", lambda: prob(a))
        post = lambda: get("post", lambda: prob(c))
        pooled = lambda: get("pooled", lambda: prob(a + c))
        ad = lambda: get("ad", lambda: lse(a + pdyn[None, :, None, None], 1))
        cd = lambda: get("cd", lambda: lse(c + pdyn[None, :, None, None], 1))
        af = lambda: get("af", lambda: lse(a + pl[None, None, :, None], 2))
        cf = lambda: get("cf", lambda: lse(c + pl[None, None, :, None], 2))
        av = lambda: get("av", lambda: lse(a + ps[:, None, None, :], 3))
        cv = lambda: get("cv", lambda: lse(c + ps[:, None, None, :], 3))
        mem = lambda: get("mem", lambda: ad() + cd() + sctx)

        def jump(j):
            def run():
                u = a + ps[:, None, None, :]
                ma = u.amax(3, keepdim=True)
                mc = c.amax(3, keepdim=True)
                v = (u - ma).exp() @ kernels[j] * (c - mc).exp()
                return v.sum(3).clamp_min(1e-35).log() + ma[:, :, :, 0] + mc[:, :, :, 0]
            return get(("s", j), run)
        f = {}
        for col in need:
            if col == "pre_cluster":
                f[col] = 1 - pre()[:, 0]
            elif col == "post_cluster":
                f[col] = 1 - post()[:, 0]
            elif col == "pooled_cluster":
                f[col] = 1 - pooled()[:, 0]
            elif col == "all":
                anypre = lse(a + prior, (1, 2, 3))
                anypost = lse(c + prior, (1, 2, 3))
                f[col] = anypre + anypost - z0
            elif col == "new_memory":
                f[col] = lse(mem(), (1, 2)) - z0
            elif col == "new_law":
                f[col] = lse(af() + cf() + dscale, (1, 2)) - z0
            elif col == "new_scale":
                f[col] = lse(av() + cv() + dctx, (1, 2)) - z0
            elif col == "memory_on":
                f[col] = lse(a[:, 0] + cd() + sctx, (1, 2)) - z0
            elif col == "memory_off":
                f[col] = lse(ad() + c[:, 0] + sctx, (1, 2)) - z0
            elif col == "memory_and_scale":
                f[col] = lse(lse(ad() + ps[:, None, :], 2) + lse(cd() + ps[:, None, :], 2) + pl, 1) - z0
            elif col == "memory_and_law":
                f[col] = lse(lse(ad() + pl[None, :, None], 1) + lse(cd() + pl[None, :, None], 1) + ps, 1) - z0
            elif col == "scale_and_law":
                f[col] = lse(lse(af() + ps[:, None, :], 2) + lse(cf() + ps[:, None, :], 2) + pdyn, 1) - z0
            elif col.startswith("scale_jump"):
                f[col] = lse(jump(int(col[-1])) + dctx, (1, 2)) - z0
            elif col.startswith("scale_cluster"):
                f[col] = lse((jump(int(col[-1])) + dctx)[:, 1:], (1, 2)) - z0
            elif col.endswith(("_decay0", "_decay1", "_decay2", "_decay3", "_decay4")):
                j = int(col[-1])
                ix = slice(1 + j * (len(MIX) - 1), 1 + (j + 1) * (len(MIX) - 1))
                f[col] = (pre() if col.startswith("pre") else post())[:, ix].sum(1)
            elif col.endswith("_null"):
                f[col] = lse(joint[:, :, int(col[3])], (1, 2)) - z0
            else:
                f[col] = lse(mem()[:, int(col[3])], 1) - z0
        arr = torch.stack([f[k] for k in need], 1).clip(-100, 160).cpu().numpy() if need else np.zeros((B, 0), np.float32)
        rows.append(pd.DataFrame(arr, index=ids[sl], columns=need))
    tab = pd.concat(rows)
    tab.index.name = "id"
    assert np.isfinite(tab).all().all()
    tab.to_parquet(path)
    print(part, "dynamic volatility (slim)", tab.shape, flush=True)


def supported(kind):
    #False when the model reads a column the slim function cannot name (predict.py then runs the original step)
    try:
        if kind == "hetero":
            hetero_plan(slim_cols.needed("hetero_jump"))
        elif kind.startswith("memory"):
            memory_plan(slim_cols.needed(f"memory_law{kind[-1]}"), int(kind[-1]))
        elif kind.startswith("native_law_"):
            native_plan(slim_cols.needed(kind))
        elif kind.startswith("latent_level_"):
            latent_plan(slim_cols.needed(kind))
        elif kind == "volatility_law":
            vol_check(slim_cols.needed(kind))
        else:
            return False
        return True
    except KeyError:
        return False


def main():
    if sys.argv[1] == "volatility_law":
        kind, part = "volatility_law", sys.argv[sys.argv.index("--part") + 1]       #mechanism.py volatility_law --part test
    else:
        kind, part = sys.argv[1], sys.argv[2]
    import ku_workspace
    root = ku_workspace.root()
    if kind == "hetero" or kind.startswith("memory"):
        sys.path.insert(0, str(root / "experiments/weight_space"))              #as white_model_features.py
        sys.argv = ["white_model_features"]
    else:
        sys.path[:0] = [str(root / "experiments/weight_space"), str(root / "experiments/v6")]      #as late_features.py / mechanism.py
        sys.argv = ["mechanism" if kind == "volatility_law" else "late_features"]
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with torch.inference_mode(), threadpool_limits(1):
        if kind == "hetero":
            import white_hetero_shift as m
            slim_hetero(m, part)
        elif kind.startswith("memory"):
            import white_memory_law as m
            slim_memory(m, part, int(kind[-1]), m.bank())
        elif kind.startswith("native_law_"):
            import white_native_law as m
            slim_native(m, part, kind.removeprefix("native_law_"))
        elif kind.startswith("latent_level_"):
            import white_latent_level as m
            slim_latent(m, kind.removeprefix("latent_level_"), part)
        elif kind == "volatility_law":
            import white_volatility_law as m
            slim_volatility(m, part)
        else:
            raise SystemExit(f"unknown kind {kind}")


if __name__ == "__main__":
    main()
