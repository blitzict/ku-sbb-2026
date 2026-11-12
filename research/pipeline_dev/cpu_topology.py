"""Select distinct physical CPU cores within the current Windows allowance.

The former benchmark chose consecutive logical processors, which can select
both SMT threads of the same core. This does not increase the CPU allowance.
GetLogicalProcessorInformation supplies the actual core masks; no numbering
pattern is assumed. More than one Windows processor group is out of scope.
"""
import ctypes as C
from ctypes import wintypes as W
import os


class ProcessorInformation(C.Structure):
    _fields_ = [('mask', C.c_size_t), ('relationship', W.DWORD),
                ('reserved', C.c_ulonglong * 2)]


def core_masks():
    if os.name != 'nt':
        raise RuntimeError('This topology query is for Windows')
    kernel = C.WinDLL('kernel32', use_last_error=True)
    function = kernel.GetLogicalProcessorInformation
    function.argtypes = [C.POINTER(ProcessorInformation), C.POINTER(W.DWORD)]
    function.restype = W.BOOL
    length = W.DWORD()
    if function(None, C.byref(length)) or C.get_last_error() != 122:
        raise C.WinError(C.get_last_error())
    if length.value % C.sizeof(ProcessorInformation):
        raise RuntimeError('Unexpected processor-information layout')
    array = (ProcessorInformation * (length.value // C.sizeof(ProcessorInformation)))()
    if not function(array, C.byref(length)):
        raise C.WinError(C.get_last_error())
    return [int(item.mask) for item in array if item.relationship == 0]


def select_processors(allowed_mask, count, cores):
    """Prefer one thread per core; then use siblings only if necessary."""
    allowed = [i for i in range(C.sizeof(C.c_size_t) * 8) if allowed_mask & (1 << i)]
    if count > len(allowed) or count < 1:
        raise ValueError('Insufficient allowed logical processors')
    masks = [int(mask) & allowed_mask for mask in cores if int(mask) & allowed_mask]
    # On hybrid Intel CPUs, SMT-capable cores are the performance cores.
    # Prefer these where available, but never assume bit numbers.
    masks.sort(key=lambda mask: (-mask.bit_count(), (mask & -mask).bit_length()))
    first = [(mask & -mask).bit_length() - 1 for mask in masks]
    if len(set(first)) != len(first):
        raise ValueError('Overlapping core masks')
    result = first + [i for i in allowed if i not in first]
    return result[:count]


if __name__ == '__main__':
    import json
    masks = core_masks()
    print(json.dumps({'core_logical_processors': [[i for i in range(64) if mask & (1 << i)] for mask in masks],
                      'six_distinct_cores': select_processors(sum(masks), 6, masks)}, indent=2))
