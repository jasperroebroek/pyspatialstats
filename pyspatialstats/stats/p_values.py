from typing import Optional

import numpy as np
from scipy.stats import t


def calculate_p_value(t_value, df, out: Optional[np.ndarray] = None) -> np.ndarray:
    p = 2 * (1 - t.cdf(np.abs(t_value), df=df))
    if out is None:
        return p
    out[:] = p
    return out
