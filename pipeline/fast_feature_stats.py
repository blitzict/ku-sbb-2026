"""Equivalent scalar statistics for the pinned CPU feature extractors.

The recorded formulas request statistics, not p-values. This module avoids
the unused KS probability calculation and SciPy's general axis/NaN wrapper
for finite one-dimensional NumPy inputs. Other inputs use SciPy unchanged.
It is scoped to feature modules: it never patches scipy.stats globally.
"""
from functools import wraps

import numpy as np
import scipy
from scipy import stats

PINNED_SCIPY = '1.18.1'


class StatisticOnly:
    __slots__ = ('statistic',)

    def __init__(self, statistic):
        self.statistic = statistic

    @property
    def pvalue(self):
        raise RuntimeError('This feature-only KS result does not calculate a p-value')


def finite_vectors(args):
    return bool(args) and all(isinstance(x, np.ndarray) and x.ndim == 1 and x.size
                             and x.dtype.kind in 'fiu' and np.isfinite(x).all() for x in args)


def ks_statistic(data1, data2, alternative='two-sided', method='auto', **kwargs):
    if (alternative != 'two-sided' or method not in {'auto', 'exact', 'asymp'}
            or kwargs or not finite_vectors((data1, data2))):
        return stats.ks_2samp(data1, data2, alternative=alternative, method=method, **kwargs)
    a, b = np.sort(data1), np.sort(data2)
    values = np.concatenate((a, b))
    dtype = np.result_type(a, b)
    if dtype.kind in 'iu':
        dtype = np.dtype(np.float64)
    # SciPy 1.18 uses linspace ECDF levels. Preserve these exact levels,
    # including their floating-point rounding, instead of dividing ranks.
    cdf_a = np.linspace(0, 1, len(a) + 1, dtype=dtype)[np.searchsorted(a, values, side='right')]
    cdf_b = np.linspace(0, 1, len(b) + 1, dtype=dtype)[np.searchsorted(b, values, side='right')]
    differences = cdf_a - cdf_b
    value = np.maximum(np.clip(-differences.min(), 0., 1.), differences.max())
    return StatisticOnly(value)


def scalar_wrapper(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if finite_vectors(args) and not {'axis', 'nan_policy', 'keepdims'} & kwargs.keys():
            return function(*args, _no_deco=True, **kwargs)
        return function(*args, **kwargs)
    return wrapped


class FeatureStats:
    """Read-through proxy; only reviewed finite-vector calls use fast paths."""
    NAMES = ('skew', 'kurtosis', 'ttest_ind', 'mannwhitneyu', 'levene', 'fligner',
             'mood', 'ansari', 'bartlett', 'cramervonmises_2samp', 'brunnermunzel',
             'differential_entropy')

    def __init__(self):
        if scipy.__version__ != PINNED_SCIPY:
            raise RuntimeError(f'Fast feature statistics require SciPy {PINNED_SCIPY}; got {scipy.__version__}')
        self.ks_2samp = ks_statistic
        for name in self.NAMES:
            function = getattr(stats, name)
            if '_axis_nan_policy' not in function.__code__.co_filename:
                raise RuntimeError(f'The installed SciPy wrapper changed: {name}')
            setattr(self, name, scalar_wrapper(function))

    def __getattr__(self, name):
        return getattr(stats, name)


def install(*modules):
    """Use only for clean/features.py and the two recorded grid extractors."""
    proxy = FeatureStats()
    for module in modules:
        if hasattr(module, 'stats'):
            module.stats = proxy
    return proxy
