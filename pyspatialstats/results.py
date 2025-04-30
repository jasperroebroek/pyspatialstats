from collections import namedtuple

GroupedCorrelationResult = namedtuple(
    'PyGroupedCorrelationResult',
    ["c", "p"],
)
GroupedLinearRegressionResult = namedtuple(
    'PyGroupedLinearRegressionResult',
    ["a", "b", 'se_a', 'se_b', 't_a', 't_b', "p_a", "p_b"]
)

StrataStatsResult = namedtuple(
    "StrataStatsResult",
    ['mean', 'std']
)
StrataCorrelationResult = namedtuple(
    "StrataCorrelationResult",
    ['c', 'p']
)
StrataLinearRegressionResult = namedtuple(
    'StrataLinearRegressionResult',
    ['a', 'b', 'se_a', 'se_b', 'p_a', 'p_b']
)
