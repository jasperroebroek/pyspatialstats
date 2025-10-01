import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression as SKLinearRegression

from pyspatialstats.stats.linear_regression import CyLinearRegression, CyRegressionResult


@pytest.fixture
def simple_data():
    rs = np.random.default_rng(42)
    n_samples = 100
    X = rs.random((n_samples, 2))
    y = 2.0 + 3.0 * X[:, 0] - 1.5 * X[:, 1] + 0.1 * rs.random(n_samples)
    return y, X


@pytest.fixture
def complex_data():
    rs = np.random.default_rng(123)
    n_samples = 500
    n_features = 5
    X = rs.random((n_samples, n_features))
    true_coef = np.array([1.5, -2.0, 0.5, 3.0, -0.8])
    y = 10.0 + X @ true_coef + 0.2 * rs.random(n_samples)
    return y, X, true_coef


def test_initialization():
    reg = CyLinearRegression(n_features=3)
    assert reg.n_features == 3
    assert reg.n_params == 4
    assert reg.count == 0


def test_initialization_invalid():
    with pytest.raises(ValueError):
        CyLinearRegression(n_features=0)
    with pytest.raises(ValueError):
        CyLinearRegression(n_features=-1)


def test_single_observation():
    reg = CyLinearRegression(n_features=2)

    reg.add(y=1.0, x=np.array([1.0, 2.0]))
    assert reg.count == 1

    reg.add(y=2.0, x=np.array([2.0, 3.0]))
    assert reg.count == 2


def test_batch_observations():
    """Test adding batch observations."""
    reg = CyLinearRegression(n_features=2)

    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    reg.add_batch(y, X)
    assert reg.count == 3


def test_dimension_mismatch():
    reg = CyLinearRegression(n_features=2)

    with pytest.raises(ValueError):
        reg.add(y=1.0, x=np.array([1.0]))  # Need 2 features

    with pytest.raises(ValueError):
        reg.add(y=1.0, x=np.array([1.0, 2.0, 3.0]))  # Too many features

    with pytest.raises(ValueError):
        reg.add_batch(y=np.array([1.0, 2.0]), X=np.array([[1.0, 1.0]]))  # Different n_samples

    with pytest.raises(ValueError):
        reg.add_batch(y=np.array([1.0]), X=np.array([[1.0]]))  # Wrong n_features


def test_reset():
    reg = CyLinearRegression(n_features=2)

    reg.add_batch(y=np.array([1.0, 2.0]), X=np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert reg.count == 2

    reg.reset()
    assert reg.count == 0


def test_merge():
    reg1 = CyLinearRegression(n_features=2)
    reg2 = CyLinearRegression(n_features=2)

    reg1.add_batch(y=np.array([1.0, 2.0]), X=np.array([[1.0, 1.0], [2.0, 2.0]]))
    reg2.add_batch(y=np.array([3.0, 4.0]), X=np.array([[3.0, 3.0], [4.0, 4.0]]))

    # Merge
    reg1.merge(reg2)
    assert reg1.count == 4

    # Original reg2 should be unchanged
    assert reg2.count == 2


def test_merge_incompatible():
    reg1 = CyLinearRegression(n_features=2)
    reg2 = CyLinearRegression(n_features=3)

    with pytest.raises(ValueError):
        reg1.merge(reg2)


def test_copy():
    reg = CyLinearRegression(n_features=2)
    reg.add_batch(y=np.array([1.0, 2.0]), X=np.array([[1.0, 1.0], [2.0, 2.0]]))

    reg_copy = reg.copy()
    assert reg_copy.count == reg.count
    assert reg_copy.n_features == reg.n_features

    reg_copy.add(y=3.0, x=np.array([3.0, 3.0]))
    assert reg_copy.count == 3
    assert reg.count == 2


def test_against_sklearn_simple(simple_data):
    """Compare against sklearn on simple data."""
    y, X = simple_data

    # Our implementation
    reg = CyLinearRegression(n_features=X.shape[1])
    reg.add_batch(y, X)
    result = reg.compute()

    # Sklearn
    sk_reg = SKLinearRegression(fit_intercept=True)
    sk_reg.fit(X, y)
    sk_coef = np.concatenate([[sk_reg.intercept_], sk_reg.coef_])
    sk_r2 = sk_reg.score(X, y)

    np.testing.assert_allclose(result.beta, sk_coef, rtol=1e-10)

    np.testing.assert_allclose(result.r_squared, sk_r2, rtol=1e-10)


def test_against_statsmodels_simple(simple_data):
    """Compare against statsmodels on simple data."""
    y, X = simple_data

    # Our implementation
    reg = CyLinearRegression(n_features=X.shape[1])
    reg.add_batch(y, X)
    result = reg.compute()

    # Statsmodels (add intercept column)
    X_with_intercept = sm.add_constant(X)
    sm_model = sm.OLS(y, X_with_intercept).fit()

    # Compare beta
    np.testing.assert_allclose(result.beta, sm_model.params, rtol=1e-10)

    # Compare standard errors
    np.testing.assert_allclose(result.beta_se, sm_model.bse, rtol=1e-8)

    # Compare R-squared
    np.testing.assert_allclose(result.r_squared, sm_model.rsquared, rtol=1e-10)


def test_against_scipy_simple(simple_data):
    y, X = simple_data
    X_single = X[:, 0:1]

    # Our implementation
    reg = CyLinearRegression(n_features=1)
    reg.add_batch(y, X_single)
    result = reg.compute()

    # Scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(X_single.flatten(), y)

    # Compare
    np.testing.assert_allclose(result.intercept, intercept, rtol=1e-10)
    np.testing.assert_allclose(result.slope[0], slope, rtol=1e-10)
    np.testing.assert_allclose(result.r_squared, r_value**2, rtol=1e-10)


def test_against_sklearn_complex(complex_data):
    """Compare against sklearn on complex data."""
    y, X, true_coef = complex_data

    # Our implementation
    reg = CyLinearRegression(n_features=X.shape[1])
    reg.add_batch(y, X)
    result = reg.compute()

    # Sklearn
    sk_reg = SKLinearRegression(fit_intercept=True)
    sk_reg.fit(X, y)
    sk_coef = np.concatenate([[sk_reg.intercept_], sk_reg.coef_])
    sk_r2 = sk_reg.score(X, y)

    # Compare
    np.testing.assert_allclose(result.beta, sk_coef, rtol=1e-10)
    np.testing.assert_allclose(result.r_squared, sk_r2, rtol=1e-10)


def test_against_statsmodels_complex(complex_data):
    """Compare against statsmodels on complex data."""
    y, X, true_coef = complex_data

    # Our implementation
    reg = CyLinearRegression(n_features=X.shape[1])
    reg.add_batch(y, X)
    result = reg.compute()

    # Statsmodels
    X_with_intercept = sm.add_constant(X)
    sm_model = sm.OLS(y, X_with_intercept).fit()

    # Compare
    np.testing.assert_allclose(result.beta, sm_model.params, rtol=1e-10)
    np.testing.assert_allclose(result.beta_se, sm_model.bse, rtol=1e-8)
    np.testing.assert_allclose(result.r_squared, sm_model.rsquared, rtol=1e-10)


# Various feature count tests
@pytest.mark.parametrize('n_features', [1, 2, 3, 5, 10, 20])
def test_various_feature_counts(n_features):
    """Test with various feature counts."""
    np.random.seed(42 + n_features)
    n_samples = max(50, n_features * 5)  # Ensure sufficient samples

    # Generate data
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = 5.0 + X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Our implementation
    reg = CyLinearRegression(n_features=n_features)
    reg.add_batch(y, X)
    result = reg.compute()

    # Sklearn comparison
    sk_reg = SKLinearRegression(fit_intercept=True)
    sk_reg.fit(X, y)
    sk_coef = np.concatenate([[sk_reg.intercept_], sk_reg.coef_])

    # Compare
    np.testing.assert_allclose(result.beta, sk_coef, rtol=1e-10)
    assert result.beta.shape == (n_features + 1,)
    assert result.beta_se.shape == (n_features + 1,)


@pytest.mark.parametrize('n_samples,n_features', [(50, 1), (100, 2), (200, 5), (500, 10), (1000, 20)])
def test_various_sample_sizes(n_samples, n_features):
    """Test with various sample sizes."""
    np.random.seed(42)

    # Generate data
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = 2.0 + X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Our implementation
    reg = CyLinearRegression(n_features=n_features)
    reg.add_batch(y, X)
    result = reg.compute()

    # Sklearn comparison
    sk_reg = SKLinearRegression(fit_intercept=True)
    sk_reg.fit(X, y)
    sk_coef = np.concatenate([[sk_reg.intercept_], sk_reg.coef_])

    # Compare
    np.testing.assert_allclose(result.beta, sk_coef, rtol=1e-10)


# Edge case tests
def test_insufficient_data():
    """Test with insufficient data points."""
    reg = CyLinearRegression(n_features=2)

    # Add only 2 observations (need > n_params = 3 for valid regression)
    reg.add(y=1.0, x=np.array([1.0, 2.0]))
    reg.add(y=2.0, x=np.array([2.0, 3.0]))

    result = reg.compute()

    # Should return invalid results (NaN or similar)
    assert np.isnan(result.r_squared) or result.degrees_of_freedom <= 0


def test_perfect_collinearity():
    """Test with perfectly collinear features."""
    reg = CyLinearRegression(n_features=2)

    # Create perfectly collinear data
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    reg.add_batch(y, X)
    result = reg.compute()

    print(result.summary())
    # Should handle gracefully (may return NaN or fail silently)
    # The exact behavior depends on your LAPACK error handling
    assert isinstance(result, CyRegressionResult)


def test_constant_y():
    """Test with constant target values. This should lead to an error in the lapack solver."""
    reg = CyLinearRegression(n_features=2)

    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([5.0, 5.0, 5.0, 5.0])  # Constant

    reg.add_batch(y, X)
    result = reg.compute()

    assert result.status == 3


def test_zero_variance_feature():
    """Test with zero-variance feature"""
    reg = CyLinearRegression(n_features=2)

    X = np.array([[2.0, 1.0], [2.0, 2.0], [2.0, 3.0], [2.0, 4.0]])  # First feature constant
    y = np.array([1.0, 2.0, 3.0, 4.0])

    reg.add_batch(y, X)
    result = reg.compute()

    # Should handle gracefully
    assert isinstance(result, CyRegressionResult)
    # No variance leads to an error in the lapack solver
    assert result.status == 3


def test_large_numbers():
    """Test with large numbers."""
    reg = CyLinearRegression(n_features=2)

    # Large scale data
    X = 1e6 * np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = 1e6 * np.array([1.0, 2.0, 3.0, 4.0])

    reg.add_batch(y, X)
    result = reg.compute()

    # Should handle large numbers without overflow
    assert not np.any(np.isinf(result.beta))
    assert not np.any(np.isnan(result.beta))


def test_small_numbers():
    """Test with very small numbers."""
    reg = CyLinearRegression(n_features=2)

    # Small scale data
    X = 1e-6 * np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = 1e-6 * np.array([1.0, 2.0, 3.0, 4.0])

    reg.add_batch(y, X)
    result = reg.compute()

    # Should handle small numbers without underflow
    assert not np.any(np.isinf(result.beta))
    assert not np.any(np.isnan(result.beta))


def test_mixed_add_methods():
    """Test mixing single and batch additions."""
    reg1 = CyLinearRegression(n_features=2)
    reg2 = CyLinearRegression(n_features=2)

    # Same data added differently
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    # Method 1: batch addition
    reg1.add_batch(y, X)

    # Method 2: single additions
    for i in range(len(y)):
        reg2.add(y[i], X[i])

    result1 = reg1.compute()
    result2 = reg2.compute()

    # Results should be identical
    np.testing.assert_allclose(result1.beta, result2.beta, rtol=1e-15)
    np.testing.assert_allclose(result1.beta_se, result2.beta_se, rtol=1e-15)
    np.testing.assert_allclose(result1.r_squared, result2.r_squared, rtol=1e-15)


# Incremental operation tests
def test_incremental_vs_batch():
    """Test that incremental addition gives same results as batch."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2.0 + X @ np.array([1.0, -0.5, 2.0]) + 0.1 * np.random.randn(100)

    # Batch method
    reg_batch = CyLinearRegression(n_features=3)
    reg_batch.add_batch(y, X)
    result_batch = reg_batch.compute()

    # Incremental method
    reg_inc = CyLinearRegression(n_features=3)
    for i in range(len(y)):
        reg_inc.add(y[i], X[i])
    result_inc = reg_inc.compute()

    # Should be identical
    np.testing.assert_allclose(result_batch.beta, result_inc.beta, rtol=1e-15)
    np.testing.assert_allclose(result_batch.beta_se, result_inc.beta_se, rtol=1e-15)
    np.testing.assert_allclose(result_batch.r_squared, result_inc.r_squared, rtol=1e-15)


def test_merge_vs_combined_batch():
    """Test that merging gives same results as combined batch."""
    np.random.seed(42)

    # Generate two datasets
    X1 = np.random.randn(50, 2)
    y1 = 1.0 + X1 @ np.array([2.0, -1.0]) + 0.1 * np.random.randn(50)

    X2 = np.random.randn(30, 2)
    y2 = 1.0 + X2 @ np.array([2.0, -1.0]) + 0.1 * np.random.randn(30)

    # Combined method
    X_combined = np.vstack([X1, X2])
    y_combined = np.concatenate([y1, y2])
    reg_combined = CyLinearRegression(n_features=2)
    reg_combined.add_batch(y_combined, X_combined)
    result_combined = reg_combined.compute()

    # Merge method
    reg1 = CyLinearRegression(n_features=2)
    reg2 = CyLinearRegression(n_features=2)
    reg1.add_batch(y1, X1)
    reg2.add_batch(y2, X2)
    reg1.merge(reg2)
    result_merged = reg1.compute()

    # Should be identical
    np.testing.assert_allclose(result_combined.beta, result_merged.beta)
    np.testing.assert_allclose(result_combined.beta_se, result_merged.beta_se)
    np.testing.assert_allclose(result_combined.r_squared, result_merged.r_squared)


def test_multiple_merges():
    """Test merging multiple regressors."""
    np.random.seed(42)

    # Create multiple small datasets
    regressors = []
    X_all = []
    y_all = []

    for i in range(5):
        X_i = np.random.randn(20, 3)
        y_i = 2.0 + X_i @ np.array([1.5, -0.8, 0.3]) + 0.1 * np.random.randn(20)

        reg_i = CyLinearRegression(n_features=3)
        reg_i.add_batch(y_i, X_i)
        regressors.append(reg_i)

        X_all.append(X_i)
        y_all.append(y_i)

    # Merge all into first regressor
    for i in range(1, len(regressors)):
        regressors[0].merge(regressors[i])

    result_merged = regressors[0].compute()

    # Compare with combined dataset
    X_combined = np.vstack(X_all)
    y_combined = np.concatenate(y_all)
    reg_combined = CyLinearRegression(n_features=3)
    reg_combined.add_batch(y_combined, X_combined)
    result_combined = reg_combined.compute()

    # Should be identical
    np.testing.assert_allclose(result_merged.beta, result_combined.beta, rtol=1e-15)
    np.testing.assert_allclose(result_merged.r_squared, result_combined.r_squared, rtol=1e-15)


# Result object tests
@pytest.fixture
def sample_result():
    """Create a sample result for testing."""
    reg = CyLinearRegression(n_features=2)
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    reg.add_batch(y, X)
    return reg.compute()


def test_result_properties(sample_result):
    """Test result properties."""
    assert hasattr(sample_result, 'beta')
    assert hasattr(sample_result, 'beta_se')
    assert hasattr(sample_result, 'r_squared')
    assert hasattr(sample_result, 'df')

    assert len(sample_result.beta) == 3  # intercept + 2 features
    assert len(sample_result.beta_se) == 3

    # Test convenience properties
    assert isinstance(sample_result.intercept, (int, float))
    assert len(sample_result.slope) == 2


def test_large_dataset():
    """Test with large dataset."""
    np.random.seed(42)
    n_samples = 10000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = 5.0 + X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Our implementation
    reg = CyLinearRegression(n_features=n_features)
    reg.add_batch(y, X)
    result = reg.compute()

    # Sklearn comparison
    sk_reg = SKLinearRegression(fit_intercept=True)
    sk_reg.fit(X, y)
    sk_coef = np.concatenate([[sk_reg.intercept_], sk_reg.coef_])

    # Should still be accurate
    np.testing.assert_allclose(result.beta, sk_coef, rtol=1e-8)


def test_many_features():
    """Test with many features."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features) * 0.1  # Small beta
    y = 2.0 + X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Our implementation
    reg = CyLinearRegression(n_features=n_features)
    reg.add_batch(y, X)
    result = reg.compute()

    # Should complete without errors
    assert len(result.beta) == n_features + 1
    assert not np.any(np.isnan(result.beta))
    assert not np.any(np.isinf(result.beta))


@pytest.mark.parametrize('calc_se', (True, False))
@pytest.mark.parametrize('calc_r2', (True, False))
def test_no_error_calculations(rs, calc_se, calc_r2):
    """Test with no errors."""
    reg = CyLinearRegression(n_features=2)
    X = rs.random(size=(4, 2))
    y = rs.random(size=4)
    reg.add_batch(y, X)

    result = reg.compute(calc_se=calc_se, calc_r2=calc_r2)

    assert result.status == 0
    assert result.calc_r2 == calc_r2
    assert result.calc_se == calc_se

    assert calc_se == ~np.all(result.beta_se == 0)
    assert calc_r2 == ~np.isnan(result.r_squared)
