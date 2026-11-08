"""Fixed-recipe CPU retraining policy, separate from the old search criteria.

Saving a fixed recipe does not mean it improved validation. The original
accepted/eligible fields remain measured results. Full-data refits use fresh
OOF inputs; no historical prediction table is imported by this module.
"""
import os
import numpy as np


def fixed():
    return os.environ.get("KU_SBB_REBUILD_FIXED") == "1"


def select(historical, newly_selected):
    return historical if fixed() else newly_selected


def export(accepted):
    return bool(accepted) or fixed()


def annotate(report):
    report["recipe_selection"] = "fixed_historical_settings" if fixed() else "original_search"
    report["export_implies_validation_improvement"] = False
    return report


def diagnostics(folder, rows):
    """Uniform diagnostic weights when no newly rebuilt shift model exists.

These are never sample weights for training the break detector. The printed
weighted AUC then equals ordinary AUC; the rebuild report discloses this.
"""
    path = folder / "adv_ptest_train.npy"
    if path.is_file():
        p = np.load(path)
        if p.shape != (rows,):
            raise ValueError("Diagnostic weights do not align with training rows")
        return p / np.clip(1 - p, .001, None)
    print("Diagnostic shift weights unavailable: weighted diagnostic uses uniform weights", flush=True)
    return np.ones(rows, dtype=float)
