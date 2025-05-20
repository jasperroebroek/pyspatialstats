from collections import namedtuple

CorrelationResult = namedtuple(
    "CorrelationResult",
    ["c", "p"],
)
StatsResult = namedtuple(
    "StatsResult",
    ["mean", "std"],
)
LinearRegressionResult = namedtuple(
    "LinearRegressionResult",
    ["a", "b", "se_a", "se_b", "t_a", "t_b", "p_a", "p_b"],
)
BootstrapMeanResult = namedtuple(
    "BootstrapMeanResult",
    ["mean", "se"],
)
