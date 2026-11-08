"""Bounded-memory rank-Gaussian preprocessing, numerically identical to the original whole-matrix code.

The original ports did, on the full training block R (about 180k x 2171 float32 for nn_var):
    ref = where(isnan(R), inf, R); srt = torch.sort(ref, 0).values.T.contiguous()      # copy + values + int64 indices
    nv = isfinite(srt).sum(1, keepdim=True).clamp(min=1).float()
    u = ((searchsorted(srt, z) + searchsorted(srt, z, right=True)) / 2 + 0.5) / (nv + 1)
    g = where(isnan(x), 0, erfinv(2 * u.clamp(1e-4, 1 - 1e-4) - 1) * 2 ** 0.5)
and KU_ENV.rank_reference saved np.sort(float32(R), 0) as 'sorted' in a compressed npz.

Here each column block is sorted on its own, so no full copy, value tensor or index tensor of R exists. Sorting is per column, so a column
block gives the same columns as the whole-matrix sort. srt differs from the torch sort only in where -0.0 and +0.0 sit among equal values and
in NaN being stored as +inf after sorting rather than before; searchsorted and the finite count cannot see either, so every transformed value
is the same. The transform is elementwise, so row chunks give the same values; it may write in place over its input.
The saved reference is the same array, written column by column in Fortran order; np.load(path)['sorted'] equals the original exactly.
Owner: Claude. Proven against the original functions by auc_lab_20261107/claude/audit/test_rank_bounded.py.
"""
import zipfile

import numpy as np
import torch

BLOCK = 64     # columns sorted at a time
CHUNK = 4096   # rows transformed at a time


def _parts(source):
    """Accept an array or a list of (array, row index or None). The list stands for np.vstack([a[i] for a, i in parts])."""
    return [(source, None)] if isinstance(source, np.ndarray) else list(source)


def rows_of(source):
    return sum(len(a) if i is None else (int(np.count_nonzero(i)) if np.asarray(i).dtype == bool else len(i)) for a, i in _parts(source))


def columns_of(source):
    return _parts(source)[0][0].shape[1]


def _take(a, i, cols=slice(None)):
    return a[:, cols] if i is None else a[i][:, cols] if cols == slice(None) else a[i, cols]


def stack(source, dtype=np.float32):
    """np.vstack(parts).astype(dtype) without the list of part copies."""
    parts = _parts(source)
    kind = np.result_type(*[a.dtype for a, _ in parts])
    out = np.empty((rows_of(parts), columns_of(parts)), dtype)
    at = 0
    for a, i in parts:
        block = _take(a, i)
        if block.dtype != kind: block = block.astype(kind)   # the dtype vstack would have produced, then the final cast
        out[at:at + len(block)] = block
        at += len(block)
        del block
    return out


def _column_block(parts, j, k, dtype):
    return np.concatenate([np.asarray(_take(a, i, slice(j, k))).astype(np.result_type(*[p.dtype for p, _ in parts])) for a, i in parts]).astype(dtype, copy=False)


def sorted_reference(source, path=None, block=BLOCK, dtype=np.float32):
    """Return (srt, nv) as the original code built them, and optionally write the KU_ENV.rank_reference npz.

    source is the training block, or its parts. dtype is the dtype the original code fed to torch (np.vstack(...).astype(dtype))."""
    parts = _parts(source)
    n, c = rows_of(parts), columns_of(parts)
    srt = torch.empty((c, n), dtype=torch.from_numpy(np.empty(0, dtype)).dtype)
    nv = torch.empty((c, 1), dtype=torch.float32)
    stream = archive = None
    if path is not None:
        archive = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True)
        stream = archive.open("sorted.npy", "w", force_zip64=True)
        np.lib.format.write_array_header_1_0(stream, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                                                      "fortran_order": True, "shape": (n, c)})
    try:
        for j in range(0, c, block):
            k = min(c, j + block)
            raw = _column_block(parts, j, k, dtype)
            ordered = np.sort(raw, 0)                                    # NaN last, after +inf
            if stream is not None:
                ref = ordered if raw.dtype == np.float32 else np.sort(raw.astype(np.float32), 0)
                stream.write(np.ascontiguousarray(ref.T).tobytes())     # Fortran order: one column after another
                del ref
            t = torch.from_numpy(ordered.T.copy())
            t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
            srt[j:k] = t
            nv[j:k] = torch.isfinite(t).sum(1, keepdim=True).clamp(min=1).float()
            del raw, ordered, t
    finally:
        if stream is not None: stream.close(); archive.close()
    return srt, nv


def transform(rows, srt, nv, out=None, chunk=CHUNK):
    """Rank-Gaussian values of rows against srt, computed by row chunks. out may be rows itself (as a tensor) for an in-place transform."""
    x = torch.as_tensor(rows)
    result = torch.empty(x.shape, dtype=torch.float32) if out is None else out
    for i in range(0, len(x), chunk):
        z = x[i:i + chunk]
        nan = torch.isnan(z)
        zs = torch.where(nan, torch.zeros_like(z), z).T.contiguous()
        u = ((torch.searchsorted(srt, zs).float() + torch.searchsorted(srt, zs, right=True).float()) / 2 + 0.5) / (nv + 1)
        g = (torch.erfinv(2 * u.clamp(1e-4, 1 - 1e-4) - 1) * 2 ** 0.5).T
        result[i:i + chunk] = torch.where(nan, torch.zeros_like(g), g)
        del z, nan, zs, u, g
    return result
