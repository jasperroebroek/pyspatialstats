import numpy as np
from scipy.stats import t


def calculate_p_value(t_value, df):
    return 2 * (1 - t.cdf(np.abs(t_value), df=df))
