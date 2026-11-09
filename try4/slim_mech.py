"""try4 fast path for the jump-law mechanism steps (pipeline_helpers/mechanism.py jump_law_<stream> --part test),
white_filtered_steps.py and white_model_features.py boundary_<stream>.

each slim_* function runs the same code as the original extract() (same tensors, same order, same batching) for only
the kernels / designs / evidence kinds that the columns the models read need, so every kept value is identical. the
columns come from the model folder in use (slim_cols.needed, KU_TRY4_MODEL).

python try4/slim_mech.py jump_law_<stream> --part test   (inside the prediction work folder)
python try4/slim_mech.py boundary_<stream> test
python try4/slim_mech.py filtered_step
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slim_cols


def jump_plan(need):
    #what extract() must build for the requested columns
    want = set()                                   #level<j> / scale<j> kernels
    shapes = set()                                 #shape branches: 0 = delta None, 1 = law shifts
    post = False
    relax = False
    for c in need:
        if re.fullmatch(r"pre\d+", c):
            post = True
        elif m := re.fullmatch(r"(level|scale)(\d)(_law\d+)?", c):
            want.add(m.group(1) + m.group(2))
            post |= m.group(3) is not None
        elif m := re.fullmatch(r"shape(\d)(_robust|_from\d+|_to\d+)?", c):
            shapes.add(int(m.group(1)))
            relax |= m.group(2) == "_robust"
        elif c == "relax":
            relax = True
        else:
            raise KeyError(c)
    if relax:
        want |= {"level0", "scale0"}               #relaxed uses the level0 / scale0 contrasts
    return want, shapes, post, relax


def slim_extract(m, part, stream):
    import torch
    need = slim_cols.needed(f"jump_law_{stream}")
    want, shapes, use_post, relax = jump_plan(need)
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
    loc_k = {j: m.jump_kernel(m.MU, np.exp(m.LOG_SD), r, w) for j, (r, w) in enumerate(levels) if f"level{j}" in want}
    scale_k = {j: m.jump_kernel(m.LOG_SD, np.ones(1), r, w) for j, (r, w) in enumerate(scales) if f"scale{j}" in want}
    core = np.array([m.noise_core(z) for z in m.SELECT])
    spacing = m.LOG_SD[1] - m.LOG_SD[0]
    shifts = np.log(core[:, None] / core[None, :]) / spacing
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
        feat, nearest = {}, {}
        if use_post:
            pp = lse(a + pm + ps, (2, 3)) + pmass
            post = pp.softmax(1)
            feat.update({f"pre{k0}": v for k0, v in enumerate(post.T)})
        for kind, kernels in (("level", loc_k), ("scale", scale_k)):
            for j, ker in kernels.items():
                if kind == "level":
                    score = m.integrated_shift(a.transpose(-1, -2), c.transpose(-1, -2), mu[:, None, None, :], sd[:, None, :], ker)
                else:
                    score = m.integrated_shift(a, c, ps, mu[:, None, :], ker)
                contrast = lse(score + pmass, 1) - z0
                feat[kind + str(j)] = contrast
                if j in (0, 2, 4):
                    for t in range(k):
                        if f"{kind}{j}_law{t}" in need:
                            feat[f"{kind}{j}_law{t}"] = post[:, t] * (score[:, t] - null[:, t]).clip(-18, 45)
                if j == 0:
                    nearest[kind] = contrast
        if relax:
            relaxed = lse(torch.stack([z0 * 0, nearest["level"], nearest["scale"]], 1) + m.tensor([0.7, 0.18, 0.12]).log(), 1)
            feat["relax"] = relaxed
        for j, delta in enumerate((None, shifts)):
            if j not in shapes:
                continue
            arr = []
            for p in range(k):
                cp = c
                if delta is not None:
                    off = delta[p, :, None] + np.arange(len(m.LOG_SD))[None, :]
                    ind = np.floor(off).astype(int).clip(0, len(m.LOG_SD) - 2)
                    index = torch.as_tensor(ind, device="cpu")[None, :, None, :].expand(B, -1, len(m.MU), -1)
                    frac = m.tensor((off - ind).clip(0, 1))[None, :, None, :]
                    cp = (1 - frac) * c.gather(-1, index) + frac * c.gather(-1, index + 1)
                    cp = cp.masked_fill(torch.as_tensor((off < 0) | (off > len(m.LOG_SD) - 1), device="cpu")[None, :, None, :], -10000)
                arr.append(lse(a[:, p:p + 1] + cp + pm + ps, (2, 3)))
            joint = torch.stack(arr, 1) + pmass[None, :, None] + pmass[None, None, :]
            mix = lse(joint, (1, 2)) - z0
            feat[f"shape{j}"] = mix
            if f"shape{j}_robust" in need:
                feat[f"shape{j}_robust"] = mix - relaxed
            for t in range(k):
                if f"shape{j}_from{t}" in need:
                    feat[f"shape{j}_from{t}"] = (lse(joint[:, t], 1) - z0).clip(-18, 45)
                if f"shape{j}_to{t}" in need:
                    feat[f"shape{j}_to{t}"] = (lse(joint[:, :, t], 1) - z0).clip(-18, 45)
        if need:
            rows.append(torch.stack([feat[c] for c in need], 1).clip(-100, 180).cpu().numpy())
        else:
            rows.append(np.zeros((B, 0), np.float32))
    tab = pd.DataFrame(np.vstack(rows), columns=need, index=pd.Index(ids, name="id"))
    assert np.isfinite(tab).all().all()
    tab.to_parquet(path)
    print(stream, part, tab.shape, "(slim)", flush=True)


FILTER_NAMES = ("null", "observed_step", "innovation_step", "reset", "short_ramp", "long_ramp", "transient", "smooth_pulse",
                "early_step", "late_step", "ar1", "ar2")


def filtered_plan(need):
    #designs (dim 1 of D) each kept column uses; design 0 (null) is always needed for the evidence baseline
    keep = {0}
    for c in need:
        base = c.rsplit("_", 1)[0] if c.endswith(("_amplitude", "_signal", "_modes")) else c
        if c == "placement":
            keep |= {1, 8, 9}
        elif c == "permanent":
            keep |= {1, 2, 4, 5, 3, 6, 7}
        elif c == "output_or_driver":
            keep |= {1, 2}
        elif base in FILTER_NAMES and FILTER_NAMES.index(base) > 0:
            keep.add(FILTER_NAMES.index(base))
        else:
            raise KeyError(c)
    return sorted(keep)


def slim_filtered(m, part="test"):
    #white_filtered_steps.extract(): regression() is computed per design (dim 1) independently, so only the designs the
    #kept columns use are fitted; evidence/amplitude/placement use the same formulas on the same values
    import torch
    need = slim_cols.needed("filtered_step")
    with np.load(m.OUT / f"pulse_streams_{part}.npz") as q:
        ids, x, n, b = [q[k] for k in ("ids", "separate", "n", "b")]
    with np.load(m.OUT / f"hetero_path_{part}.npz") as q:
        assert np.array_equal(ids, q["ids"]) and np.array_equal(n - 2, q["n"])
        hh = q["common"]
    ar = pd.read_parquet(m.OUT / f"state_{part}.parquet").loc[ids, ["ar1", "ar2"]].to_numpy()
    bank = m.emissions()
    mass = m.tensor(np.log(m.LAW_MASS))
    keep = filtered_plan(need)
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
        rel = ll - ll[:, 0:1]
        lp = ll[:, pos[0]] + mass
        w = lp.softmax(1)
        zero = torch.logsumexp(lp, 1)
        evidence = torch.logsumexp(ll + mass, 2) - zero[:, None]
        e = {k: evidence[:, pos[k]] for k in keep}
        rec = {}
        for c in need:
            if c == "placement":
                rec[c] = e[1] - torch.logsumexp(torch.stack([e[8], e[9]], 1), 1) + np.log(2)
            elif c == "permanent":
                rec[c] = torch.logsumexp(torch.stack([e[1], e[2], e[4], e[5]], 1), 1) - torch.logsumexp(torch.stack([e[3], e[6], e[7]], 1), 1) + np.log(0.75)
            elif c == "output_or_driver":
                rec[c] = e[1] - e[2]
            elif c.endswith("_amplitude"):
                kk = pos[FILTER_NAMES.index(c[:-10])]
                rec[c] = (w * amp[:, kk].abs()).sum(1)
            elif c.endswith("_signal"):
                kk = pos[FILTER_NAMES.index(c[:-7])]
                rec[c] = (w * info[:, kk].sqrt() * amp[:, kk].abs()).sum(1)
            elif c.endswith("_modes"):
                kk = pos[FILTER_NAMES.index(c[:-6])]
                rec[c] = (w[:, 3:] * rel[:, kk, 3:].clamp(-20, 60)).sum(1)
            else:
                rec[c] = e[FILTER_NAMES.index(c)]
        arr = torch.stack([rec[c] for c in need], 1).clamp(-80, 110).cpu().numpy() if need else np.zeros((B, 0), np.float32)
        rows.append(pd.DataFrame(arr, columns=need, index=ids[sl]))
    frame = pd.concat(rows)
    frame.index.name = "id"
    assert np.isfinite(frame).all().all()
    frame.to_parquet(m.OUT / f"filtered_step_{part}.parquet")
    print(part, "filtered steps (slim)", frame.shape, flush=True)


BOUNDARY_KINDS = ("level", "scale", "shape", "any", "scale_shape", "level_shape")


def boundary_kind(c):
    for kd in sorted(BOUNDARY_KINDS, key=len, reverse=True):
        if c.startswith(kd + "_"):
            return kd
    raise KeyError(c)


def slim_boundary(m, part, stream):
    #white_boundary_law.extract(): evidence() for only the kinds the kept columns use (same expressions, same order)
    import torch
    need = slim_cols.needed(f"boundary_law_{stream}")
    kinds = [kd for kd in BOUNDARY_KINDS if any(boundary_kind(c) == kd for c in need)]

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
        if not kinds:
            blocks.append(pd.DataFrame(index=ids[start:end]))
            continue
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


def supported(kind):
    #False when the model reads a column the slim function cannot name (predict.py then runs the original step)
    try:
        if kind.startswith("jump_law_"):
            jump_plan(slim_cols.needed(kind))
        elif kind.startswith("boundary_"):
            [boundary_kind(c) for c in slim_cols.needed("boundary_law_" + kind.removeprefix("boundary_"))]
        elif kind == "filtered_step":
            filtered_plan(slim_cols.needed("filtered_step"))
        return True
    except KeyError:
        return False


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
