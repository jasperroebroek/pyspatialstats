import numpy as np
from sklearn.linear_model import LinearRegression

from pyspatialstats.random.random import Random
from pyspatialstats.results.stats import MeanResult, RegressionResult
from pyspatialstats.stats.p_values import calculate_p_value


def np_bootstrap_mean(v, n_bootstraps, seed=None, rng=None):
    if rng is None:
        rng = Random(seed)
    means = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        sample_idx = rng.np_randints(bound=v.size, n=v.size)
        sample = v[sample_idx]
        means[i] = np.mean(sample)

    return MeanResult(mean=means.mean(), se=means.std())


def np_bootstrap_linear_regression(x, y, n_bootstraps, seed=None):
    if n_bootstraps < 2:
        raise ValueError('Bootstrap sample size must be at least 2')
    if x.shape[0] < 2:
        raise ValueError('At least 2 samples must be present')
    if y.shape[0] != x.shape[0]:
        raise ValueError('x and y must have the same number of rows')

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    nf = x.shape[1] + 1
    betas = np.zeros((n_bootstraps, nf))
    r2s = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        x_sample, y_sample = x[idx], y[idx]
        model = LinearRegression().fit(x_sample, y_sample)
        betas[i, 1:] = model.coef_
        betas[i, 0] = model.intercept_
        r2s[i] = model.score(x_sample, y_sample)

    df = n - nf
    beta = betas.mean(axis=0)
    beta_se = betas.std(axis=0, ddof=1)
    t = beta / beta_se
    p_values = calculate_p_value(t, df)

    return RegressionResult(
        df=df,
        beta=beta,
        beta_se=beta_se,
        r_squared=r2s.mean(),
        r_squared_se=r2s.std(ddof=1),
        t=t,
        p=p_values,
    )
