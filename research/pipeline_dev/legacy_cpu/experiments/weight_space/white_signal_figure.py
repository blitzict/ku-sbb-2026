# CPU port; original experiments/weight_space/white_signal_figure.py, SHA256 648e44ac691226b64f1acbe3eaf78f6cba91263a3bd8e3fddd9c9cc7169a2592
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Plot an observed volatility change and two-mode distribution from the historical cohort."""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from partition_evidence import ROOT, OUT
import white_loop as wl

def main():
    _, ids, _, _, _, _, miss = wl.load()
    m = pd.read_parquet(OUT / 'mode_residual_train.parquet').loc[ids]
    group = m.loc[miss & (m.gate > 0.8)]
    chosen = group.occupancy_change.abs().idxmax()
    case = '00025ef9b811b1466360f7abe81d5bf6'
    with np.load(OUT / 'pulse_streams_train.npz') as a:
        assert np.array_equal(a['ids'], ids)
        i, j = ids.get_indexer([case, chosen])
        series = a['separate']
        n, b = (a['n'], a['b'])
        first = series[i, :n[i]].astype(float)
        second = series[j, :n[j]].astype(float)
        cut1, cut2 = (b[i], b[j])
    ar = pd.read_parquet(OUT / 'state_train.parquet', columns=['ar1', 'ar2']).loc[case].to_numpy()
    noise = first[2:] - np.column_stack([first[1:-1], first[:-2]]) @ ar
    cut1 -= 2
    noise -= np.r_[np.full(cut1, np.median(noise[:cut1])), np.full(len(noise) - cut1, np.median(noise[cut1:]))]
    t = np.arange(len(noise)) - cut1
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.5, 6.0))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.85, bottom=0.15, hspace=0.85)
    ink = '#19272e'
    blue = '#216477'
    orange = '#a74420'
    fig.patch.set_facecolor('#ffffff')
    for ax in (top, bottom):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#98a5a8')
        ax.tick_params(colors=ink, labelsize=9)
        ax.grid(axis='y', color='#e3e8e8', lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle('Two signals in real missed breaks', ha='left', x=0.105, y=0.98, color=ink, fontsize=18, fontweight='semibold')
    top.set_title('Large moves cluster before the boundary', loc='left', color=ink, fontsize=12, pad=8)
    top.plot(t, abs(noise), color='#a9b8bc', lw=0.72, label='Absolute innovation')
    for k, (tt, u) in enumerate(((t[:cut1], noise[:cut1]), (t[cut1:], noise[cut1:]))):
        rms = np.sqrt(np.convolve(u * u, np.ones(12) / 12, mode='valid'))
        top.plot(tt[11:], rms, color=blue, lw=1.8, label='12-observation RMS' if k == 0 else None)
    top.axvline(0, color=orange, lw=1.15, ls='--', label='Marked boundary')
    top.set(xlabel='Observations relative to the boundary', ylabel='Magnitude (series units)', xlim=(t.min(), t.max()), ylim=(0, None))
    top.legend(loc='upper right', ncol=3, frameon=False, fontsize=8, handlelength=2)
    bottom.set_title('Two common noise levels before and after the boundary', loc='left', color=ink, fontsize=12, pad=8)
    bins = np.linspace(second.min() - 0.001, second.max() + 0.001, 27)
    for u, color, style, label in ((second[:cut2], blue, '-', 'Before'), (second[cut2:], orange, '--', 'After')):
        counts = np.histogram(u, bins)[0]
        bottom.stairs(counts / len(u), bins, color=color, linestyle=style, linewidth=2, label=label)
    bottom.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bottom.yaxis.set_major_locator(MaxNLocator(4))
    bottom.set(xlabel='Observation after removing short pulses (series units)', ylabel='Share per bin', ylim=(0, None))
    bottom.legend(frameon=False, ncol=2, loc='upper right', fontsize=9)
    fig.text(0.105, 0.027, 'Selected examples from the 821 historical misses. Cohort-wide gains are reported separately.', color='#44575c', fontsize=9)
    out = ROOT / 'new weights/white_noise_signal_examples.png'
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print('volatility example', case, 'two-mode example', chosen, 'eligible', len(group), 'selected', group.loc[chosen].to_dict(), flush=True)
if __name__ == '__main__':
    main()
