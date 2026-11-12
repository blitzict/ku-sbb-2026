"""Per-series CPU detectors for the raw-data trainer.

Only numerical functions from the original project are imported. Each receives
one complete competition series and its supplied boundary. No function here
opens a prediction archive, reads labels, or fits a cross-series transform.
"""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, gammaln, logsumexp

SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "weight_space"
sys.path.insert(0, str(SOURCE))
import curve_likelihood
import harmonic_change
import innovation_transform
import partition_evidence
import periodic_components
import phase_likelihood
import pulse_filter
import state_filter
import trend_clock
import trend_components


# Use finer histogram bins in the bulk, where small location shifts matter,
# and wider bins in the tails. Interpolate observations between adjacent bins.
TAIL = np.geomspace(6.2, 60, 18)
AXIS = np.r_[-TAIL[::-1], np.linspace(-6, 6, 97), TAIL]
MEANS = np.linspace(-1.65, 1.65, 25)
LOG_SCALES = np.linspace(-1.05, 1.05, 21)
FAMILIES = ("gaussian", "student", "contamination", "two_peak")


@lru_cache(maxsize=1)
def density_bank():
    """Precompute normalized log densities for a compact location/scale grid."""
    z = (AXIS[:, None, None] - MEANS[None, :, None]) / np.exp(LOG_SCALES)
    normal = lambda x: -.5 * (x**2 + np.log(2 * np.pi))
    laws = [normal(z)]
    for df in (4, 7, 12):
        laws.append(gammaln((df+1)/2) - gammaln(df/2) - .5*np.log(np.pi*(df-2))
                    - .5*(df+1)*np.log1p(z**2/(df-2)))
    for mass, ratio in ((.025, 4.5), (.08, 2.25)):
        sd = 1 / np.sqrt(1-mass + mass*ratio**2)
        laws.append(np.logaddexp(np.log1p(-mass) + normal(z/sd) - np.log(sd),
                                 np.log(mass) + normal(z/(sd*ratio)) - np.log(sd*ratio)))
    for gap in (.65, .8, .9, .95):
        width = np.sqrt(1-gap**2)
        laws.append(np.logaddexp(normal((z-gap)/width), normal((z+gap)/width)) - np.log(2*width))
    values = np.stack(laws, axis=1) - LOG_SCALES
    return np.ascontiguousarray(values.reshape(len(AXIS), -1), dtype=np.float32)


def density_features(stream, boundary):
    """Compare shared vs changed location, scale or noise law on both sides."""
    center = np.median(stream)
    scale = max(1.4826*np.median(abs(stream-center)), .18*stream.std(), .01)
    z = (stream-center)/scale
    counts = []
    for segment in (z[:boundary], z[boundary:]):
        idx = np.searchsorted(AXIS, segment).clip(1, len(AXIS)-1) - 1
        right = ((segment-AXIS[idx])/(AXIS[idx+1]-AXIS[idx])).clip(0, 1)
        counts.append(np.bincount(idx, 1-right, minlength=len(AXIS))
                      + np.bincount(idx+1, right, minlength=len(AXIS)))
    ll = (np.array(counts, dtype=np.float32) @ density_bank()).astype(float)
    ll -= ll.max(axis=1, keepdims=True)
    a, b = ll.reshape(2, -1, len(MEANS), len(LOG_SCALES))
    # Equal prior mass per family, divided among its parameter settings.
    family = np.repeat(np.arange(4), [1, 3, 2, 4])
    prior = -np.log(4*np.bincount(family)[family])
    mean_penalty, scale_penalty = np.log(len(MEANS)), np.log(len(LOG_SCALES))
    shared = logsumexp(a+b, axis=(1, 2)) - mean_penalty - scale_penalty
    location = logsumexp(logsumexp(a, axis=1)+logsumexp(b, axis=1), axis=1) - 2*mean_penalty - scale_penalty
    spread = logsumexp(logsumexp(a, axis=2)+logsumexp(b, axis=2), axis=1) - mean_penalty - 2*scale_penalty
    null = logsumexp(shared+prior)
    pa = logsumexp(a, axis=(1, 2)) - mean_penalty - scale_penalty
    pb = logsumexp(b, axis=(1, 2)) - mean_penalty - scale_penalty
    law_change = logsumexp(logsumexp(a+prior[:, None, None], axis=0)
                           + logsumexp(b+prior[:, None, None], axis=0)) - mean_penalty - scale_penalty
    result = dict(location=logsumexp(location+prior)-null, scale=logsumexp(spread+prior)-null,
                  shape=law_change-null, joint=logsumexp(pa+prior)+logsumexp(pb+prior)-null,
                  center=abs(center), reference_scale=scale)
    for k, name in enumerate(FAMILIES):
        mask = family == k
        for label, vec in (("before", pa), ("after", pb), ("shared", shared)):
            result[name+"_"+label] = logsumexp((vec+prior)[mask])-logsumexp(vec+prior)
        result[name+"_location"] = logsumexp((location+prior)[mask])-null
        result[name+"_scale"] = logsumexp((spread+prior)[mask])-null
    return result


def measure(job):
    """Return named measurements. The ID is only used to align the output."""
    ident, raw, cut = job
    a, b = raw[:cut], raw[cut:]
    rows = {"length_before": len(a), "length_after": len(b)}

    def add(prefix, data):
        rows.update({prefix+"_"+key: value for key, value in data.items()})

    add("harmonic", dict(zip(harmonic_change.NAMES, harmonic_change.score(a, b), strict=True)))
    common, separate, wave = periodic_components.residuals(a, b)
    add("wave", wave)
    season = wave["gate"]*expit((wave["period"]-9)/1.5)*expit((38-wave["period"])/2)
    _, _, memory = innovation_transform.transform(raw, cut, common, separate, season)
    add("memory", memory)
    if wave["gate"] > .025 and 8 <= wave["period"] <= 49:
        add("phase", phase_likelihood.one(raw, cut, wave["period"]))
    _, _, trend = trend_components.one(raw, cut, memory["pre_walk"], season)
    add("trend", trend)
    keep = (1-memory["pre_walk"])*(1-season)
    _, curve, curve_info = curve_likelihood.one(raw, cut, trend["phi"], trend["df"])
    add("curve", curve_info)
    corrected = raw + keep*(curve-raw)
    corrected += season*(separate-corrected)
    pulse_clean = raw
    if min(len(a), len(b)) > 42 and np.var(corrected) > 1e-14:
        pulse_clean, corrected, pulse = pulse_filter.one(raw, corrected, cut)
        add("pulse", pulse)
    _, _, clock = trend_clock.one(pulse_clean, cut, np.clip(trend["phi"], -.92, .94), trend["df"], keep)
    add("clock", clock)
    common_state, separate_state, state = state_filter.state(corrected, cut, np.clip(trend["df"], 4, 60))
    add("state", state)
    names = (view+"_"+stat for view in partition_evidence.VIEWS for stat in partition_evidence.STATS)
    add("rank", dict(zip(names, partition_evidence.row_features(raw, cut), strict=True)))
    for name, values in (("observed", raw), ("corrected", corrected), ("innovation", separate_state)):
        add("density_"+name, density_features(values, cut))
    add("state_gaps", innovation_transform.moments(common_state[2:cut], common_state[cut:]))
    if not np.isfinite(list(rows.values())).all():
        raise ValueError(f"Non-finite detector output for {ident}")
    return ident, rows
