import numpy as np
from scipy.stats import pearsonr
from statsmodels import api as sm

from pyspatialstats.results.stats import CorrelationResult, RegressionResult
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


def focal_linear_regression_simple(a1, a2, window=5):
    if a2.ndim == 2:
        a2 = np.expand_dims(a2, axis=2)
    nf = a2.shape[2] + 1

    a1[np.isnan(a2).sum(axis=2) > 0] = np.nan

    window = define_window(window)
    mask = window.get_mask(2)
    fringes = window.get_fringes(False, 2)

    df = np.full(a1.shape, 0, dtype=np.uintp)
    beta = np.full(a1.shape + (nf,), np.nan)
    beta_se = np.full(a1.shape + (nf,), np.nan)
    t = np.full(a1.shape + (nf,), np.nan)
    p = np.full(a1.shape + (nf,), np.nan)
    r_squared = np.full(a1.shape, np.nan)

    for i in range(fringes[0], a1.shape[0] - fringes[0]):
        for j in range(fringes[1], a1.shape[1] - fringes[1]):
            ind = np.s_[i - fringes[0] : i + fringes[0] + 1, j - fringes[1] : j + fringes[1] + 1]

            if np.isnan(a1[i, j]):
                continue

            d1 = a1[ind][mask]
            d2 = a2[ind][mask]

            values_mask = ~np.isnan(d1)
            d1 = d1[values_mask]
            d2 = d2[values_mask]

            # statsmodels implementation
            d1_with_intercept = sm.add_constant(d1)
            model = sm.OLS(d2, d1_with_intercept)
            result = model.fit()

            df[i, j] = result.df_resid
            beta[i, j] = result.params
            beta_se[i, j] = result.bse
            t[i, j] = result.tvalues
            p[i, j] = result.pvalues
            r_squared[i, j] = result.rsquared

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
