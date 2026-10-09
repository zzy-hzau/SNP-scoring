"""Array fields and empirical percentile calculations."""
import numpy as np
import pandas as pd

def _column(frame, names):
    lower = {str(c).lower(): c for c in frame.columns}
    return next((lower[n.lower()] for n in names if n.lower() in lower), None)

def _boolean(frame, names, default=False):
    col = _column(frame, names)
    if not col: return np.full(len(frame), default, dtype=bool)
    return frame[col].fillna(False).astype(str).str.lower().isin({"true", "1", "1.0", "yes", "valid", "passed", "direct", "core"}).to_numpy(copy=True)

def midrank_percentile(values, valid=None):
    """Frozen paper calibration: (average rank - 0.5)/number of cores."""
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(x)
    out = np.full(len(x), np.nan)
    if ok.any(): out[ok] = (pd.Series(x[ok]).rank(method="average").to_numpy() - .5) / ok.sum()
    return out
