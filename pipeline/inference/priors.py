"""Apply learned noise-law priors to new series, without estimating priors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch

from .common import require, save_table

WS = "tune_out/weight_space"


def outer(model_root, work, seed=0, tag="physical", view=None):
    from white_law_head import unpack
    model_root, work = Path(model_root), Path(work)
    folder = f"law_outer{seed}" + ("" if tag == "physical" else f"_{tag}")
    query_folder = work / WS if view is None else work / WS / f"view_{view}"
    with np.load(query_folder / f"wnlaw_{tag}_test.npz") as data:
        ids, values = data["ids"].astype(str), data["values"].astype(float)
    for fold in range(5):
        params = json.loads(require(model_root / WS / folder / f"fold{fold}_prior.json").read_text())
        _, likelihood, features = unpack(values, np.asarray(params["law"]))
        odds = (logsumexp(likelihood + np.log(params["change"]), axis=1)
                - logsumexp(likelihood + np.log(params["null"]), axis=1))
        array = np.column_stack([odds, features]).clip(-80, 80)
        table = pd.DataFrame(array, index=pd.Index(ids, name="id"),
                             columns=[f"v{i}" for i in range(array.shape[1])])
        save_table(query_folder / folder / f"fold{fold}_test.parquet", table)


def transport(model_root, work, stream="state"):
    import white_population_transport as module
    model_root, work = Path(model_root), Path(work)
    # prepare() computes deterministic per-series likelihoods, not a fitted population.
    with np.load(module.prepare(stream, "test")) as data:
        ids, index, before, after = [data[k] for k in ("ids", "i", "a", "b")]
    white = pd.read_parquet(work / WS / "innov_v2_test.parquet").loc[ids, "pre_white"].to_numpy()
    predictions = []
    with torch.inference_mode():
        for fold in range(5):
            with np.load(require(model_root / WS / f"population_transport_{stream}_models/fold{fold}.npz")) as params:
                latent, small = module.odds(index, before, after, white, params["pop"])
                extra = []
                for temperature, rates in zip((0.45, 1.0), params["rates"]):
                    scores = temperature * latent
                    extra.append(logsumexp(scores + np.log(rates[1]), axis=1)
                                 - logsumexp(scores + np.log(rates[0]), axis=1))
                predictions.append(np.column_stack([small, *extra]))
    array = np.mean(predictions, axis=0)
    table = pd.DataFrame(array, index=pd.Index(ids, name="id"),
                         columns=[f"v{i}" for i in range(array.shape[1])])
    save_table(work / WS / f"transport_{stream}_test.parquet", table)


def transitions(model_root, work):
    import white_transition_priors as module
    model_root, work = Path(model_root), Path(work)
    likelihoods = {}
    ids = None
    for stream in ("state", "pulse"):
        table = pd.read_parquet(work / WS / f"native_law_{stream}_test.parquet")
        if ids is not None and not ids.equals(table.index):
            raise ValueError("Native-law feature identifiers differ")
        ids = table.index
        for reference in ("native", "standard"):
            likelihoods[stream + "_" + reference] = module.chart(table, reference)[0]
    white = pd.read_parquet(work / WS / "innov_v2_test.parquet").loc[ids, "pre_white"].to_numpy()
    gates = module.tensor(np.stack([white, 1 - white], axis=1))
    predictions = []
    columns = []
    with torch.inference_mode():
        for fold in range(5):
            fields = {}
            with np.load(require(model_root / WS / f"transition_prior_models/fold{fold}.npz")) as params:
                for key, likelihood in likelihoods.items():
                    for number, temperature in enumerate((0.4, 0.94)):
                        evidence = torch.exp((likelihood - likelihood.amax(1, keepdim=True)) * temperature).clamp_min(1e-30)
                        name = key + str(number)
                        coefficients = module.tensor(params[name])
                        null, change = [(gates * (evidence @ weight.T)).sum(1).clamp_min(1e-29).log()
                                        for weight in coefficients]
                        fields[name] = (change - null).cpu().numpy()
            table = pd.DataFrame(fields, index=ids)
            columns = table.columns
            predictions.append(table.to_numpy())
    save_table(work / WS / "transition_odds_test.parquet",
               pd.DataFrame(np.mean(predictions, axis=0), index=ids, columns=columns))
