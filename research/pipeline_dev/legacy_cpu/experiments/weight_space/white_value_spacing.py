# CPU port; original experiments/weight_space/white_value_spacing.py, SHA256 06ac32981ee77df3fadcebad9de2243bfc826ff1e7443da89b5ffb24079ab194
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether observations repeat at regular intervals along the value axis."""
import contextlib as _cpu_context
import os as _cpu_os
from pathlib import Path as _CpuPath

def _cpu_workspace():
    value = _cpu_os.environ.get('KU_SBB_ROOT')
    if not value:
        raise RuntimeError('Set KU_SBB_ROOT to the fresh CPU rebuild workspace')
    root = _CpuPath(value).resolve()
    if not (root / 'CPU_REBUILD_WORKSPACE.json').is_file():
        raise RuntimeError('CPU ports require a marked rebuild workspace')
    return root
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
FREQUENCIES = np.geomspace(2.6, 64, 48).astype(np.float32)

def measure(values, boundary):
    left, right = (values[:boundary], values[boundary:])
    centered = np.r_[left - np.median(left), right - np.median(right)]
    scale = max(np.quantile(abs(centered), 0.68), 0.15 * centered.std(), 0.001)
    phase = (values - np.median(values))[:, None] * FREQUENCIES / scale
    waves = np.cos(phase) + 1j * np.sin(phase)
    segments = (waves[:boundary], waves[boundary:])
    means = []
    sizes = []
    for segment in segments:
        block = np.arange(len(segment)) // 4 % 2
        means.append(np.stack([segment[block == k].mean(0) for k in (0, 1)]))
        sizes.append([sum(block == k) for k in (0, 1)])
    means = np.asarray(means)
    sizes = np.asarray(sizes, dtype=float)
    result = {}
    for label, band in (('two_peaks', FREQUENCIES < 7), ('fine_steps', FREQUENCIES >= 7)):
        votes = []
        for select, check in ((0, 1), (1, 0)):
            discovery = means[:, select, band]
            power = abs(discovery) ** 2 * sizes[:, select, None]
            j = np.max(power, axis=0).argmax()
            selected_side = power[:, j].argmax()
            held = means[:, check, band][:, j]
            agreement = (discovery[:, j].conj() * held).real
            confidence = np.sqrt(sizes[selected_side, check]) * agreement[selected_side] / max(abs(discovery[selected_side, j]), 0.02)
            gate = expit((confidence - 3.5) / 0.6)
            change = held[1] - held[0]
            discovery_change = discovery[1, j] - discovery[0, j]
            joint_n = 1 / np.sum(1 / sizes[:, check])
            magnitude = (abs(held[1]) - abs(held[0])) * np.sqrt(joint_n)
            direction = (change * discovery_change.conj()).real * joint_n
            votes.append([confidence, gate * magnitude, gate * abs(magnitude), gate * np.arcsinh(direction), gate * (agreement[1] - agreement[0])])
        result.update(zip([label + '_' + s for s in ('repeats', 'magnitude', 'abs_magnitude', 'change', 'strength')], np.mean(votes, axis=0)))
    return result

def extract(part):
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, values, lengths, bounds = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    rows = [measure(v[:n], b) for v, n, b in zip(values, lengths, bounds)]
    table = pd.DataFrame(rows, index=pd.Index(ids, name='id'))
    assert np.isfinite(table).all().all()
    table.to_parquet(OUT / f'value_spacing_{part}.parquet')
    print(part, len(table), table.filter(like='repeats').gt(4).sum().to_dict(), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for partition in ('train', 'test'):
            extract(partition)
