"""try3 fast path for the jump-law mechanism steps (pipeline_helpers/mechanism.py jump_law_<stream> --part test).

white_jump_priors.extract() builds 149 columns from 12 integrated-shift kernels; try3 reads 2-7 of them per stream.
slim_extract() runs the same code (same tensors, same order) for only the kernels and columns the models read, so
every kept value is identical.

python try3/slim_mech.py jump_law_<stream> --part test   (inside the prediction work folder)
"""
import sys

import numpy as np
import pandas as pd

NEED = {
    "nullstate": ["scale1", "scale3", "shape0", "shape0_from4", "shape0_robust", "shape0_to3", "shape0_to4"],
    "prestate": ["shape0", "shape0_to4"],
    "pulse": ["level5", "shape0"],
    "smoothvol": ["scale2_law7", "scale3", "scale5", "shape0"],
    "state": ["level5", "shape0", "shape0_from4"],
}


def slim_extract(m, part, stream):
    import torch
    need = NEED[stream]
    path = m.OUT / f"jump_law_{stream}_{part}.parquet"
    with np.load(m.OUT / f"{stream}_streams_{part}.npz") as q:
        ids, x, n, b = (q[k] for k in ("ids", "separate", "n", "b"))
    k = len(m.SELECT)
    idx = [m.LAWS.index(z) for z in m.SELECT]
    full = m.dictionary()
    bank = full.reshape(len(full), len(m.LAWS), len(m.MU), len(m.LOG_SD))[:, idx].flatten(1).contiguous()
    del full
    pmass = m.tensor(m.MASS).log()
    levels = [(0.0, 0.09), (0.0, 0.22), (0.0, 0.55), (0.0, 1.2), (0.36, 0.15), (1.1, 0.32)]
    scales = [(0.0, 0.045), (0.0, 0.12), (0.0, 0.3), (0.0, 0.6), (0.3, 0.11), (0.54, 0.16)]
    robust = any(c.endswith("_robust") for c in need)
    #kernels needed: the named level/scale columns (scaleJ_lawT uses scaleJ), plus level0/scale0 for the robust shape
    want = {c.split("_")[0] for c in need if c.startswith(("level", "scale"))}
    if robust:
        want |= {"level0", "scale0"}
    loc_k = {j: m.jump_kernel(m.MU, np.exp(m.LOG_SD), r, w) for j, (r, w) in enumerate(levels) if f"level{j}" in want}
    scale_k = {j: m.jump_kernel(m.LOG_SD, np.ones(1), r, w) for j, (r, w) in enumerate(scales) if f"scale{j}" in want}
    rows = []
    for start in range(0, len(x), 40):
        zz, center, noise = [], [], []
        for r, j, e in zip(x[start:start + 40], b[start:start + 40], n[start:start + 40]):
            raw = r[:e].astype(float)
            md = np.median(raw)
            s = max(np.median(abs(raw - md)) * 1.4826, 0.18 * raw.std(), 0.01)
            z = (raw - md) / s
            zz.append([m.hist(z[:j]), m.hist(z[j:])])
            center.append(md)
            noise.append(s)
        B = len(zz)
        ll = (m.tensor(np.array(zz)).reshape(2 * B, -1) @ bank).reshape(B, 2, k, len(m.MU), len(m.LOG_SD))
        ll -= ll.amax((2, 3, 4), keepdims=True)
        a, c = ll.unbind(1)
        mu, sd = m.priors(np.array(center), np.array(noise))
        pm = mu[:, None, :, None]
        ps = sd[:, None, None, :]
        lse = torch.logsumexp
        null = lse(a + c + pm + ps, (2, 3))
        z0 = lse(null + pmass, 1)
        pp = lse(a + pm + ps, (2, 3)) + pmass
        post = pp.softmax(1)
        feat, nearest = {}, {}
        for kind, kernels in (("level", loc_k), ("scale", scale_k)):
            for j, ker in kernels.items():
                if kind == "level":
                    score = m.integrated_shift(a.transpose(-1, -2), c.transpose(-1, -2), mu[:, None, None, :], sd[:, None, :], ker)
                else:
                    score = m.integrated_shift(a, c, ps, mu[:, None, :], ker)
                contrast = lse(score + pmass, 1) - z0
                feat[kind + str(j)] = contrast
                for t in range(k):
                    if f"{kind}{j}_law{t}" in need:
                        feat[f"{kind}{j}_law{t}"] = post[:, t] * (score[:, t] - null[:, t]).clip(-18, 45)
                if j == 0:
                    nearest[kind] = contrast
        arr = [lse(a[:, p:p + 1] + c + pm + ps, (2, 3)) for p in range(k)]       #shape0: the delta=None branch
        joint = torch.stack(arr, 1) + pmass[None, :, None] + pmass[None, None, :]
        mix = lse(joint, (1, 2)) - z0
        feat["shape0"] = mix
        if robust:
            relaxed = lse(torch.stack([z0 * 0, nearest["level"], nearest["scale"]], 1) + m.tensor([0.7, 0.18, 0.12]).log(), 1)
            feat["shape0_robust"] = mix - relaxed
        for t in range(k):
            if f"shape0_from{t}" in need:
                feat[f"shape0_from{t}"] = (lse(joint[:, t], 1) - z0).clip(-18, 45)
            if f"shape0_to{t}" in need:
                feat[f"shape0_to{t}"] = (lse(joint[:, :, t], 1) - z0).clip(-18, 45)
        rows.append(torch.stack([feat[c] for c in need], 1).clip(-100, 180).cpu().numpy())
    tab = pd.DataFrame(np.vstack(rows), columns=need, index=pd.Index(ids, name="id"))
    assert np.isfinite(tab).all().all()
    tab.to_parquet(path)
    print(stream, part, tab.shape, "(slim)", flush=True)


FILTERED = (0, 1, 4, 8, 9, 10)      #null, observed_step, short_ramp, early_step, late_step, ar1: all that ar1/early_step/placement/short_ramp need


def slim_filtered(m, part="test"):
    #white_filtered_steps.extract(): regression() is computed per design (dim 1) independently, so only the designs the
    #kept columns use are fitted; evidence/placement use the same formulas on the same values
    import torch
    with np.load(m.OUT / f"pulse_streams_{part}.npz") as q:
        ids, x, n, b = [q[k] for k in ("ids", "separate", "n", "b")]
    with np.load(m.OUT / f"hetero_path_{part}.npz") as q:
        assert np.array_equal(ids, q["ids"]) and np.array_equal(n - 2, q["n"])
        hh = q["common"]
    ar = pd.read_parquet(m.OUT / f"state_{part}.parquet").loc[ids, ["ar1", "ar2"]].to_numpy()
    bank = m.emissions()
    mass = m.tensor(np.log(m.LAW_MASS))
    keep = list(FILTERED)
    pos = {k: i for i, k in enumerate(keep)}
    rows = []
    for start in range(0, len(n), 30):
        sl = slice(start, start + 30)
        B = len(x[sl])
        T = int(n[sl].max()) - 2
        obs = np.zeros((B, T)); h = np.ones_like(obs); ok = np.zeros_like(obs)
        D = np.zeros((B, len(keep), T))
        for i, (row, end, j, phi, height) in enumerate(zip(x[sl], n[sl], b[sl], ar[sl], hh[sl])):
            L = end - 2
            u = row[2:end] - phi[0] * row[1:end - 1] - phi[1] * row[:end - 2]
            s = max(1.4826 * np.median(abs(u - np.median(u))), 0.18 * np.std(u), 0.015)
            obs[i, :L] = (u - np.median(u)) / s
            h[i, :L] = height[:L]
            ok[i, :L] = 1
            D[i, :, :L] = m.designs(row, end, j, phi, s)[keep]
        ll, amp, info = m.regression(m.tensor(obs), m.tensor(h), m.tensor(D), m.tensor(ok), bank)
        assert torch.isfinite(ll).all()
        lp = ll[:, pos[0]] + mass
        zero = torch.logsumexp(lp, 1)
        evidence = torch.logsumexp(ll + mass, 2) - zero[:, None]
        e = {k: evidence[:, pos[k]] for k in keep}
        rec = {"ar1": e[10], "early_step": e[8],
               "placement": e[1] - torch.logsumexp(torch.stack([e[8], e[9]], 1), 1) + np.log(2), "short_ramp": e[4]}
        arr = torch.stack(list(rec.values()), 1).clamp(-80, 110).cpu().numpy()
        rows.append(pd.DataFrame(arr, columns=list(rec), index=ids[sl]))
    frame = pd.concat(rows)
    frame.index.name = "id"
    assert np.isfinite(frame).all().all()
    frame.to_parquet(m.OUT / f"filtered_step_{part}.parquet")
    print(part, "filtered steps (slim)", frame.shape, flush=True)


BOUNDARY = {
    "pulse": ["any_at", "any_far_weak", "any_odds0.35", "level_shape_at", "level_shape_long_strong", "level_shape_odds0.35",
              "level_shape_odds1.0", "scale_shape_three_weak", "shape_at"],
    "state": ["any_at", "any_far_strong", "any_far_weak", "any_long_strong", "any_long_weak", "any_mid_weak", "any_odds0.35",
              "any_short_weak", "scale_shape_far_weak"],
}


def slim_boundary(m, part, stream):
    #white_boundary_law.extract(): evidence() for only the kinds the kept columns use (same expressions, same order)
    import torch
    need = BOUNDARY[stream]
    kinds = {"pulse": ["shape", "any", "scale_shape", "level_shape"], "state": ["any", "scale_shape"]}[stream]

    def evidence(left, right, mu, sd, mass):
        lse = torch.logsumexp
        pm = mu[:, None, None, :, None]
        ps = sd[:, None, None, None, :]
        w = mass[None, None, :, None, None]
        pooled = lse(left + right + pm + ps + w, (2, 3, 4))
        out = {}
        if "level" in kinds:
            lm, rm = (lse(t + pm, 3) for t in (left, right))
            out["level"] = lse(lm + rm + mass[None, None, :, None] + sd[:, None, None, :], (2, 3))
        if "scale" in kinds:
            ls, rs = (lse(t + ps, 4) for t in (left, right))
            out["scale"] = lse(ls + rs + mass[None, None, :, None] + mu[:, None, None, :], (2, 3))
        if {"shape", "scale_shape", "level_shape"} & set(kinds):
            lf, rf = (lse(t + w, 2) for t in (left, right))
        if "shape" in kinds:
            out["shape"] = lse(lf + rf + mu[:, None, :, None] + sd[:, None, None, :], (2, 3))
        if "any" in kinds:
            a, b = (lse(t + pm + ps + w, (2, 3, 4)) for t in (left, right))
            out["any"] = a + b
        if "scale_shape" in kinds:
            out["scale_shape"] = lse(lse(lf + sd[:, None, None, :], 3) + lse(rf + sd[:, None, None, :], 3) + mu[:, None, :], 2)
        if "level_shape" in kinds:
            out["level_shape"] = lse(lse(lf + mu[:, None, :, None], 2) + lse(rf + mu[:, None, :, None], 2) + sd[:, None, :], 2)
        return torch.stack([out[k] for k in kinds], -1) - pooled[:, :, None]

    with np.load(m.OUT / f"{stream}_streams_{part}.npz") as d:
        ids, n, b, x = (d[k] for k in ("ids", "n", "b", "separate"))
    bank = m.dictionary()
    mass = m.tensor(m.W).log()
    M, S = m.M, m.S
    blocks = []
    for start in range(0, len(x), 24):
        end = min(len(x), start + 24)
        group = [m.prepare(r[:l].astype(float), j) for r, l, j in zip(x[start:end], n[start:end], b[start:end])]
        counts, md, s = map(np.asarray, zip(*group))
        size = len(md)
        ll = (m.tensor(counts).reshape(-1, len(m.AXIS)) @ bank).reshape(size, -1, len(m.W), len(M), len(S))
        left = ll[:, :-1]
        right = ll[:, -1:] - left
        left = left - left.amax((2, 3, 4), keepdim=True)
        right = right - right.amax((2, 3, 4), keepdim=True)
        absolute = md[:, None] + s[:, None] * M
        scale = np.log(s[:, None]) + S
        mu = m.tensor(np.log(0.89 / 4 / (1 + np.exp(((abs(absolute) - 2) / 0.075).clip(-35, 35))) + 0.11 * np.exp(-abs(absolute) / 2) / 4) + np.log(s[:, None] * np.diff(M)[0]))
        sd = m.tensor(np.log(0.88 / 1.4 / (1 + np.exp(((abs(scale) - 0.7) / 0.045).clip(-35, 35))) + 0.12 * np.exp(-abs(scale)) * 0.5) + np.log(np.diff(S)[0]))
        v = evidence(left, right, mu, sd, mass).cpu().numpy()
        center = v[:, 7, :]
        features = {}
        for j, name in enumerate(kinds):
            h = v[:, :, j]
            c = center[:, j]
            features[name + "_at"] = c
            for offset, tag in ((1, "one"), (2, "three"), (3, "near"), (4, "short"), (5, "mid"), (6, "long"), (7, "far")):
                dl = c - h[:, 7 - offset]
                dr = c - h[:, 7 + offset]
                features[name + "_" + tag + "_weak"] = np.minimum(dl, dr)
                features[name + "_" + tag + "_strong"] = np.maximum(dl, dr)
            for t in (0.35, 1.0):
                rel = (h - c[:, None]) * t
                features[name + f"_odds{t}"] = -np.logaddexp.reduce(rel[:, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]], axis=1) + np.log(10)
        blocks.append(pd.DataFrame({c: features[c] for c in need}, index=ids[start:end]).clip(-140, 240).astype(np.float32))
    df = pd.concat(blocks)
    df.index.name = "id"
    assert np.isfinite(df).all().all()
    df.to_parquet(m.OUT / f"boundary_law_{stream}_{part}.parquet")
    print(part, stream, df.shape, "(slim)", flush=True)


def main():
    if sys.argv[1].startswith("boundary_"):
        import ku_workspace
        root = ku_workspace.root()
        sys.path.insert(0, str(root / "experiments/weight_space"))
        stream, part = sys.argv[1].removeprefix("boundary_"), sys.argv[2]
        sys.argv = ["white_model_features"]
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(1)
        import white_boundary_law as m
        with torch.inference_mode(), threadpool_limits(1):
            slim_boundary(m, part, stream)
        return
    if sys.argv[1] == "filtered_step":
        import ku_workspace
        root = ku_workspace.root()
        sys.path[:0] = [str(root / "experiments/weight_space"), str(root / "experiments/v6")]
        sys.argv = ["query-features"]
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(1)
        import white_filtered_steps as m
        with threadpool_limits(limits=1), torch.inference_mode():
            slim_filtered(m)
        return
    kind, part = sys.argv[1], sys.argv[sys.argv.index("--part") + 1]
    import ku_workspace
    root = ku_workspace.root()
    sys.path[:0] = [str(root / "experiments/weight_space"), str(root / "experiments/v6")]
    sys.argv = ["mechanism"]
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    import white_jump_priors as m
    with threadpool_limits(limits=1), torch.inference_mode():
        slim_extract(m, part, kind.removeprefix("jump_law_"))


if __name__ == "__main__":
    main()
