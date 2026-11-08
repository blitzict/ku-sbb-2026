"""CPU segment statistics, in small batches, with the original fixed formulas.

This is the feature-producing part of v6/generator_probe.py. It retains the
1024-sample padding, AR orders 0..16, 1e-4 ridge, and Student-t grid. The original
AIC diagnostic used the effective length at order 16 for every candidate; that
choice is retained because p_aic is a recorded model input. The Student-t
normalizing constant uses the independently verified float64 correction.

Labels are written only as metadata and to the separate diagnostic delta table.
No label is used by measure(), and test labels are unavailable.
"""
import argparse
from time import monotonic

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import ku_workspace as env

LENGTH = 1024
MAX_ORDER = 16


def measure(segments):
    """Return all historical numeric fields for a bounded batch of segments."""
    count = len(segments)
    n = torch.tensor([len(s) for s in segments])
    if not count or (n <= MAX_ORDER).any() or (n > LENGTH).any():
        raise ValueError("Segment lengths must lie between 17 and 1024")
    x = torch.zeros(count, LENGTH)
    for i, segment in enumerate(segments):
        x[i, :len(segment)] = torch.as_tensor(segment, dtype=torch.float32)
    time = torch.arange(LENGTH)[None, :]
    valid = (time < n[:, None]).float()
    mean = (x * valid).sum(1) / n
    sd = ((((x - mean[:, None]) * valid) ** 2).sum(1) / n).sqrt()
    if (sd == 0).any():
        raise ValueError("A constant segment cannot use the original AR normalization")
    z = (x - mean[:, None]) / sd[:, None] * valid
    fits, bic = {}, []
    for order in range(MAX_ORDER + 1):
        if order == 0:
            beta, residual, effective = torch.zeros(count, 1), z * valid, n.float()
        else:
            windows = z.unfold(1, order + 1, 1)
            response = windows[:, :, -1]
            design = torch.cat([windows[:, :, :-1].flip(-1), torch.ones_like(response)[..., None]], -1)
            mask = (time[:, order:] < n[:, None]).float()[..., None]
            weighted = design * mask
            gram = weighted.transpose(1, 2) @ design + 1e-4 * torch.eye(order + 1)
            rhs = weighted.transpose(1, 2) @ response[..., None]
            beta = torch.linalg.solve(gram, rhs).squeeze(-1)
            residual = torch.zeros(count, LENGTH)
            residual[:, order:] = (response - (design @ beta[..., None]).squeeze(-1)) * mask.squeeze(-1)
            effective = (n - order).float()
        variance = residual.square().sum(1) / effective
        bic.append(effective * variance.log() + (order + 1) * effective.log())
        fits[order] = beta, residual, effective, variance
    chosen = torch.stack(bic, 1).argmin(1)
    # Deliberately retain the recorded historical AIC formula (see docstring).
    aic = torch.stack([effective * fits[p][3].log() + 2 * (p + 1) for p in fits], 1)
    fields = dict(p_bic=chosen, p_aic=aic.argmin(1))
    b4 = fits[4][0]
    fields.update({f"phi{j+1}": b4[:, j] for j in range(4)})
    fields.update(phi1_ar1=fits[1][0][:, 0], persist4=b4[:, :4].sum(1),
                  innov_sd4=fits[4][3].sqrt() * sd, mean=mean, sd=sd, n=n.float())
    r = torch.stack([fits[p][1] for p in fits], 1)[torch.arange(count), chosen]
    effective = n.float() - chosen.float()
    mask = (time >= chosen[:, None]).float() * valid
    r = r / (r.square().sum(1) / effective).sqrt()[:, None] * mask
    fields.update(skew=r.pow(3).sum(1) / effective, kurt=r.pow(4).sum(1) / effective - 3,
                  tail3=((r.abs() > 3).float() * mask).sum(1) / effective,
                  tail2=((r.abs() > 2).float() * mask).sum(1) / effective)
    padded = r.masked_fill(mask == 0, float("nan")).nan_to_num(0)
    q = lambda p: torch.quantile(padded, p, dim=1)
    fields["q_ratio"] = (q(.99) - q(.01)) / (q(.75) - q(.25) + 1e-6)

    def acf(values, lag):
        return (values[:, lag:] * values[:, :-lag] * mask[:, lag:] * mask[:, :-lag]).sum(1) / (values.square() * mask).sum(1)

    for lag in (1, 2, 3, 5, 10):
        fields[f"racf{lag}"] = acf(r, lag)
    squared = (r.square() - 1) * mask
    for lag in (1, 2, 5):
        fields[f"sqacf{lag}"] = acf(squared, lag)
    dfs = torch.tensor([2.1, 2.3, 2.6, 3., 3.5, 4., 5., 6., 8., 10., 15., 25., 50., 200.])
    likelihoods = []
    for df in dfs:
        scale = ((df - 2) / df).sqrt()
        standardized = r / scale
        logp = env.t_constant(df, standardized) - scale.log() - (df + 1) / 2 * torch.log1p(standardized.square() / df)
        likelihoods.append((logp * mask).sum(1) / effective)
    likelihoods = torch.stack(likelihoods, 1)
    gaussian = (-.5 * np.log(2 * np.pi) - .5 * r.square()) * mask
    fields.update(t_df=dfs[likelihoods.argmax(1)], ll_t_minus_gauss=likelihoods.max(1).values - gaussian.sum(1) / effective)
    spectrum = torch.fft.rfft(z * valid, n=LENGTH, dim=1).abs().square()
    spectrum[:, 0] = 0
    fields.update(spec_peak_share=spectrum.max(1).values / spectrum.sum(1), spec_peak_freq=spectrum.argmax(1).float() / LENGTH)
    half = (time < (n[:, None] // 2)).float() * valid
    first = (z * half).sum(1) / half.sum(1)
    second = (z * (valid - half)).sum(1) / (valid - half).sum(1)
    fields["half_mean_diff"] = (second - first) * sd / fields["innov_sd4"]
    return pd.DataFrame({key: value.float().numpy() for key, value in fields.items()})


def deltas(table):
    """Reproduce the separate diagnostic table; it is not a feature input."""
    pre = table[(table.part == "pre") & table.train].set_index("id")
    post = table[(table.part == "post") & table.train].set_index("id").loc[pre.index]
    d = pd.DataFrame(index=pre.index)
    d["d_mean"] = (post["mean"] - pre["mean"]) / pre.innov_sd4
    d["d_logsd_innov"] = np.log(post.innov_sd4 / pre.innov_sd4)
    d["d_logsd_level"] = np.log(post.sd / pre.sd)
    d["d_phi1"] = post.phi1 - pre.phi1
    d["d_persist"] = post.persist4 - pre.persist4
    d["d_phi_l2"] = np.sqrt(sum((post[f"phi{j}"] - pre[f"phi{j}"]) ** 2 for j in (1, 2, 3, 4)))
    d["d_kurt"] = post["kurt"] - pre["kurt"]
    d["d_skew"] = post["skew"] - pre["skew"]
    d["d_logdf"] = np.log(post.t_df) - np.log(pre.t_df)
    d["d_tail3"] = post.tail3 - pre.tail3
    d["d_qratio"] = np.log(post.q_ratio / pre.q_ratio)
    d["d_sqacf1"] = post.sqacf1 - pre.sqacf1
    d["post_half_drift"] = post.half_mean_diff
    d["pre_half_drift"] = pre.half_mean_diff
    d["label"] = pre.label
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.batch <= 256:
        parser.error("--batch must be between 1 and 256 for the memory target")
    root = env.root()
    labels = pd.read_csv(root / "data/train_labels.csv", dtype={"id": str}).set_index("id").label
    results = []
    start = monotonic()
    torch.set_num_threads(1)
    with threadpool_limits(1), torch.inference_mode():
        for split in ("train", "test"):
            # Dictionary IDs keep the raw frame small; only one split is held.
            raw = env.read_series(root / f"data/{split}_series.parquet")
            rows, meta = [], []

            def flush():
                if rows:
                    values = measure(rows)
                    info = pd.DataFrame(meta, columns=["id", "part", "n_pre", "n_post"])
                    info["train"] = split == "train"
                    info["label"] = info.id.map(labels).fillna(-1).astype(int)
                    results.append(pd.concat([info, values], axis=1))
                    rows.clear()
                    meta.clear()

            for i, (ident, frame) in enumerate(raw.groupby("id", sort=False, observed=True)):
                v = frame.value.to_numpy(np.float32)
                boundary = int(frame.period.to_numpy().argmax())
                for side, segment in (("pre", v[:boundary]), ("post", v[boundary:])):
                    rows.append(segment)
                    meta.append((str(ident), side, boundary, len(v) - boundary))
                    if len(rows) == args.batch:
                        flush()
                if (i + 1) % 2000 == 0:
                    print(f"Segments {split}: {i+1} series, {monotonic()-start:.0f}s", flush=True)
            flush()
            del raw
    table = pd.concat(results, ignore_index=True)
    if not np.isfinite(table.select_dtypes(include="number")).all().all():
        raise ValueError("Non-finite generator features")
    table.to_parquet(root / "tune_out/generator_segments.parquet")
    deltas(table).to_parquet(root / "tune_out/generator_deltas.parquet")
    print(f"Wrote {len(table)} segment rows, {monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
