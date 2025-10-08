import numpy as np
from scipy.stats import pearsonr
from statsmodels import api as sm

from pyspatialstats.results.stats import CorrelationResult, RegressionResult
from pyspatialstats.rolling import rolling_window
from pyspatialstats.stats.p_values import calculate_p_value
from pyspatialstats.windows import define_window


def overlapping_arrays(a1, a2, preserve_input=True):
    a1 = np.asarray(a1, dtype=np.float64, copy=preserve_input)
    a2 = np.asarray(a2, dtype=np.float64, copy=preserve_input)

    if a1.shape != a2.shape or a1.ndim != 2:
        raise ValueError('arrays are not of the same size or not 2d')

    invalid_cells = np.logical_or(np.isnan(a1), np.isnan(a2))
    a1[invalid_cells] = np.nan
    a2[invalid_cells] = np.nan

    return a1, a2


def focal_linear_regression_simple(x, y, window=5, reduce=False):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=2)
    nf = x.shape[2] + 1

    if x.ndim != 3 or y.ndim != 2:
        raise ValueError('x must be 3d and y must be 2d')

    y[np.isnan(x).sum(axis=2) > 0] = np.nan

    window = define_window(window)
    mask = window.get_mask(2)
    fringe = window.get_fringes(reduce=reduce, ndim=2)

    shape = window.define_windowed_shape(reduce=reduce, a=y)

    df = np.full(shape, np.nan)
    beta = np.full(shape + (nf,), np.nan)
    beta_se = np.full(shape + (nf,), np.nan)
    t = np.full(shape + (nf,), np.nan)
    p = np.full(shape + (nf,), np.nan)
    r_squared = np.full(shape, np.nan)

    window_shape_2d = window.get_shape(ndim=2)
    window_shape_3d = window_shape_2d + (nf - 1,)

    y_windowed = rolling_window(y, window=window_shape_2d, reduce=reduce)
    x_windowed = rolling_window(x, window=window_shape_3d, reduce=reduce)

    idx_center = (mask.shape[0] // 2, mask.shape[1] // 2)

    for i in range(y_windowed.shape[0]):
        for j in range(y_windowed.shape[1]):
            if np.isnan(y_windowed[i, j, idx_center[0], idx_center[1]]) and not reduce:
                continue

            cy = y_windowed[i, j][mask]
            cx = x_windowed[i, j, 0][mask]

            values_mask = ~np.isnan(cy)
            cy = cy[values_mask]
            cx = cx[values_mask]

            # statsmodels implementation
            cx_with_intercept = sm.add_constant(cx)
            model = sm.OLS(cy, cx_with_intercept)
            result = model.fit()

            idx = (fringe[0] + i, fringe[1] + j)
            df[idx] = result.df_resid
            beta[idx] = result.params
            beta_se[idx] = result.bse
            t[idx] = result.tvalues
            p[idx] = result.pvalues
            r_squared[idx] = result.rsquared

    return RegressionResult(
        df=df,
        beta=beta,
        beta_se=beta_se,
        t=t,
        p=p,
        r_squared=r_squared,
    )


def focal_correlation_simple(a1, a2, window, fraction_accepted=0.7):
    a1, a2 = overlapping_arrays(a1, a2)

    window = define_window(window)
    mask = window.get_mask(2)
    fringes = window.get_fringes(False, 2)

    corr = np.full(a1.shape, np.nan)
    df = np.full(a1.shape, 0)

    for i in range(fringes[0], a1.shape[0] - fringes[0]):
        for j in range(fringes[1], a1.shape[1] - fringes[1]):
            ind = np.s_[i - fringes[0] : i + fringes[0] + 1, j - fringes[1] : j + fringes[1] + 1]

            if np.isnan(a1[i, j]) or np.isnan(a2[i, j]):
                continue

            d1 = a1[ind][mask]
            d2 = a2[ind][mask]

            d1 = d1[~np.isnan(d1)]
            d2 = d2[~np.isnan(d2)]

            if d1.size < fraction_accepted * mask.sum():
                continue

            if np.all(d1 == d1[0]) or np.all(d2 == d2[0]):
                corr[i, j] = 0
                continue

            corr[i, j] = pearsonr(d1, d2)[0]
            df[i, j] = d1.size - 1

    t = corr * np.sqrt(df) / np.sqrt(1 - corr**2)

    return CorrelationResult(c=corr, p=calculate_p_value(t, df))
