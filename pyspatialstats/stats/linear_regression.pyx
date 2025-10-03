# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as cnp
import numpy as np
from libc.math cimport sqrt
from libc.string cimport memset, memcpy
from numpy.math cimport NAN
from libc.stdlib cimport calloc, malloc, free
from scipy.linalg.cython_lapack cimport dpotrf, dpotrs, dpotri
from scipy.linalg.cython_blas cimport ddot

from pyspatialstats.stats.welford cimport WelfordState, ws_array_new, ws_init, ws_add, ws_mean, ws_std

# =============================================================================
# LinearRegressionState Functions
# =============================================================================

cdef inline LinearRegressionState* lrs_new(size_t nf) noexcept nogil:
    """
    Create and initialize a new LinearRegressionState.
    Returns NULL on failure.
    """
    cdef LinearRegressionState* lrs = <LinearRegressionState*> malloc(sizeof(LinearRegressionState))
    if lrs_init(lrs, nf) != 0:
        free(lrs)
        return NULL
    return lrs


cdef inline int lrs_init(LinearRegressionState* lrs, size_t nf) noexcept nogil:
    """
    Initialize an existing LinearRegressionState with given nf.
    Returns 0 on success, -1 on failure.
    """
    if lrs == NULL:
        return -1

    lrs.nf = nf
    lrs.XtX = <double*> malloc(nf * nf * sizeof(double))
    lrs.Xty = <double*> malloc(nf * sizeof(double))

    if lrs.XtX == NULL or lrs.Xty == NULL:
        if lrs.XtX != NULL:
            free(lrs.XtX)
            lrs.XtX = NULL
        if lrs.Xty != NULL:
            free(lrs.Xty)
            lrs.Xty = NULL
        return -1

    lrs_reset(lrs)
    return 0


cdef inline void lrs_free(LinearRegressionState* lrs) noexcept nogil:
    """
    Free memory allocated for LinearRegressionState.

    Parameters:
    -----------
    lrs : LinearRegressionState*
        Pointer to state to free (can be NULL)
    """
    if lrs == NULL:
        return
    if lrs.XtX != NULL:
        free(lrs.XtX)
        lrs.XtX = NULL
    if lrs.Xty != NULL:
        free(lrs.Xty)
        lrs.Xty = NULL
    free(lrs)


cdef inline void lrs_reset(LinearRegressionState* lrs) noexcept nogil:
    """
    Reset LinearRegressionState to initial state (all zeros). nf is required to be set before calling this function and 
    must match the size of the arrays.

    Parameters:
    -----------
    lrs : LinearRegressionState*
        Pointer to state to reset
    """
    lrs.count = 0
    lrs.yty = 0.0
    lrs.y_sum = 0.0
    memset(lrs.XtX, 0, lrs.nf * lrs.nf * sizeof(double))
    memset(lrs.Xty, 0, lrs.nf * sizeof(double))


cdef inline void lrs_add(LinearRegressionState* lrs, double y, double[:] x, double weight = 1.0) noexcept nogil:
    """
    Add a single observation to the regression state with a given weight.

    Parameters:
    -----------
    lrs : LinearRegressionState*
        Pointer to regression state
    y : double
        Target value
    x : double[:]
        Feature vector (excluding intercept)
    weight : double, default 1.0
        Weight for this observation
    """
    cdef:
        size_t i, j
        double xi, xj
        double yw

    if weight <= 0.0:
        return

    yw = y * weight
    lrs.count += weight
    lrs.yty += y * yw           # y^2 * weight
    lrs.y_sum += yw             # sum(y * weight)

    # Build X'X and X'y matrices
    # First feature is intercept (always 1.0)
    for i in range(lrs.nf):
        xi = 1.0 if i == 0 else x[i - 1]
        xi *= weight
        # X'X matrix
        for j in range(i, lrs.nf):
            xj = 1.0 if j == 0 else x[j - 1]
            lrs.XtX[i * lrs.nf + j] += xi * xj
        # X'y vector
        lrs.Xty[i] += xi * y


cdef inline void lrs_merge(LinearRegressionState* lrs_into, LinearRegressionState* lrs_from) noexcept nogil:
    """
    Merge one LinearRegressionState into another.

    Parameters:
    -----------
    lrs_into : LinearRegressionState*
        Destination state (will be modified)
    lrs_from : LinearRegressionState*
        Source state (will not be modified)
    """
    if lrs_into.nf != lrs_from.nf:
        return

    cdef size_t i

    lrs_into.count += lrs_from.count
    lrs_into.yty += lrs_from.yty
    lrs_into.y_sum += lrs_from.y_sum

    for i in range(lrs_into.nf * lrs_into.nf):
        lrs_into.XtX[i] += lrs_from.XtX[i]

    for i in range(lrs_into.nf):
        lrs_into.Xty[i] += lrs_from.Xty[i]


cdef void lrs_to_result(LinearRegressionState* lrs, LinearRegressionResult* result, bint calc_se = True, bint calc_r2 = True) noexcept nogil:
    """
    Compute regression results from accumulated state.

    Parameters:
    -----------
    lrs : LinearRegressionState*
        Source regression state
    result : RegressionResult*
        Destination result structure (must be pre-allocated)
    calc_se : bint
        Whether to calculate standard errors
    calc_r2 : bint
        Whether to calculate R-squared
    """
    result.status = 0
    result.calc_se = calc_se
    result.calc_r2 = calc_r2

    if lrs.count <= lrs.nf:
        lrr_reset(result)
        result.status = 1
        return

    cdef:
        # I am filling the upper triangle of the matrix in C order -> lower triangle in Fortran/LAPACK
        char UPLO_LOWER = b'L'
        int INC_ONE = 1
        int NRHS_ONE = 1
        int info = 0

        double* XtX_copy = <double*> malloc(lrs.nf * lrs.nf * sizeof(double))
        double* Xty_copy = <double*> malloc(lrs.nf * sizeof(double))

        double y_mean, yhat_ss, rss, mse, tss
        size_t i

        int nf_int = lrs.nf

    if XtX_copy == NULL or Xty_copy == NULL:
        free(XtX_copy)
        free(Xty_copy)
        lrr_reset(result)
        result.status = 2
        return

    # Copy data since LAPACK routines modify input
    memcpy(XtX_copy, lrs.XtX, lrs.nf * lrs.nf * sizeof(double))
    memcpy(Xty_copy, lrs.Xty, lrs.nf * sizeof(double))

    result.df = <ssize_t> lrs.count - <ssize_t> lrs.nf

    # Cholesky decomposition of X'X
    dpotrf(&UPLO_LOWER, &nf_int, XtX_copy, &nf_int, &info)
    if info != 0:
        lrr_reset(result)
        result.status = 3
        free(XtX_copy)
        free(Xty_copy)
        return

    # Solve for coefficients: beta = (X'X)^(-1) X'y
    dpotrs(&UPLO_LOWER, &nf_int, &NRHS_ONE, XtX_copy, &nf_int, Xty_copy, &nf_int, &info)
    if info != 0:
        lrr_reset(result)
        result.status = 4
        free(XtX_copy)
        free(Xty_copy)
        return

    # Copy coefficients
    memcpy(result.beta, Xty_copy, lrs.nf * sizeof(double))

    if calc_se or calc_r2:
        yhat_ss = ddot(&nf_int, result.beta, &INC_ONE, lrs.Xty, &INC_ONE)
        rss = lrs.yty - yhat_ss
        mse = rss / result.df

    if calc_se:
        # Compute covariance matrix for standard errors
        dpotri(&UPLO_LOWER, &nf_int, XtX_copy, &nf_int, &info)
        if info != 0:
            result.status = 5
            free(XtX_copy)
            free(Xty_copy)
            return

        # Standard errors from diagonal of covariance matrix
        for i in range(lrs.nf):
            result.beta_se[i] = sqrt(XtX_copy[i * lrs.nf + i] * mse)

    if calc_r2:
        # R-squared
        y_mean = lrs.y_sum / lrs.count
        tss = lrs.yty - lrs.count * y_mean * y_mean
        result.r_squared = 1.0 - rss / tss if tss > 0 else 0.0

    free(XtX_copy)
    free(Xty_copy)


cdef inline LinearRegressionState* lrs_array_new(size_t count, size_t nf) noexcept nogil:
    """
    Create an array of LinearRegressionState structs and initialises them.

    Parameters:
    -----------
    count : size_t
        Number of states to allocate
    nf : size_t
        Number of features for each state (including intercept)

    Returns:
    --------
    LinearRegressionState* : Pointer to array, or NULL on failure
    """
    if count == 0:
        return NULL
    cdef:
        LinearRegressionState* lrs_array = <LinearRegressionState*> malloc(count * sizeof(LinearRegressionState))
        size_t i
        bint info
    if lrs_array == NULL:
        return NULL
    info = lrs_array_init(lrs_array, count, nf)
    if info >= 0:
        lrs_array_free(lrs_array, info)
        return NULL
    return lrs_array


cdef inline int lrs_array_init(LinearRegressionState* lrs_array, size_t count, size_t nf) noexcept nogil:
    cdef:
        size_t i
        bint info
    for i in range(count):
        info = lrs_init(&lrs_array[i], nf)
        if info != 0:
            return i + 1
    return 0


cdef inline void lrs_array_free(LinearRegressionState* lrs_array, size_t count) noexcept nogil:
    """
    Free an array of LinearRegressionState structs.
    Safe to call on NULL pointer.
    """
    if lrs_array == NULL:
        return
    cdef size_t i
    for i in range(count):
        if lrs_array[i].XtX != NULL:
            free(lrs_array[i].XtX)
            lrs_array[i].XtX = NULL
        if lrs_array[i].Xty != NULL:
            free(lrs_array[i].Xty)
            lrs_array[i].Xty = NULL
    free(lrs_array)


cdef void lrs_array_to_bootstrap_result(
    LinearRegressionState* lrs_array,
    LinearRegressionResult* lrr,
    size_t n_boot
) noexcept nogil:
    cdef:
        size_t i, j
        WelfordState r2_ws
        WelfordState* beta_ws
        LinearRegressionResult* lrr_tmp = lrr_new(lrr.nf)

    if lrs_array == NULL or n_boot == 0 or lrr == NULL or lrr_tmp == NULL:
        return

    beta_ws = ws_array_new(lrr.nf)
    if beta_ws == NULL:
        return
    ws_init(&r2_ws)

    for i in range(n_boot):
        lrs_to_result(&lrs_array[i], lrr_tmp, calc_se=False, calc_r2=True)

        # Memory allocation failed
        if lrr_tmp.status == 2:
            lrr_reset(lrr)
            lrr.status = lrr_tmp.status
            free(lrr_tmp)
            free(beta_ws)
            return

        # Model failed to converge or too few observations
        if lrr_tmp.status > 0 and lrr_tmp.status != 2:
            continue

        lrr.df += lrr_tmp.df
        ws_add(&r2_ws, lrr_tmp.r_squared)
        for j in range(lrr.nf):
            ws_add(&beta_ws[j], lrr_tmp.beta[j])

    for j in range(lrr.nf):
        lrr.beta[j] = ws_mean(&beta_ws[j])
        lrr.beta_se[j] = ws_std(&beta_ws[j])
    lrr.r_squared = ws_mean(&r2_ws)
    lrr.r_squared_se = ws_std(&r2_ws)

    if r2_ws.count == 0:
        lrr.df = -1
    else:
        lrr.df /= r2_ws.count

    free(beta_ws)
    free(lrr_tmp)


# =============================================================================
# CyRegressionResult Functions
# =============================================================================

cdef inline LinearRegressionResult* lrr_new(size_t nf) noexcept nogil:
    """
    Allocate a new RegressionResult and initialize it.
    Returns NULL on failure.
    """
    cdef LinearRegressionResult* lrr = <LinearRegressionResult*> malloc(sizeof(LinearRegressionResult))
    if lrr == NULL:
        return NULL

    if lrr_init(lrr, nf) != 0:
        free(lrr)
        return NULL

    return lrr


cdef inline int lrr_init(LinearRegressionResult* lrr, size_t nf) noexcept nogil:
    """
    Initialize a LinearRegressionResult with nf features.
    Returns 0 on success, -1 on failure.
    """
    if lrr == NULL:
        return -1

    lrr.status = 0
    lrr.nf = nf
    lrr.beta = <double*> malloc(nf * sizeof(double))
    lrr.beta_se = <double*> malloc(nf * sizeof(double))

    if lrr.beta == NULL or lrr.beta_se == NULL:
        if lrr.beta != NULL:
            free(lrr.beta)
            lrr.beta = NULL
        if lrr.beta_se != NULL:
            free(lrr.beta_se)
            lrr.beta_se = NULL
        return -1

    lrr_reset(lrr)
    return 0


cdef inline void lrr_free(LinearRegressionResult* lrr) noexcept nogil:
    """
    Free a LinearRegressionResult
    """
    if lrr == NULL:
        return
    if lrr.beta != NULL:
        free(lrr.beta)
        lrr.beta = NULL
    if lrr.beta_se != NULL:
        free(lrr.beta_se)
        lrr.beta_se = NULL
    free(lrr)


cdef inline void lrr_reset(LinearRegressionResult* lrr) noexcept nogil:
    """
    Reset LinearRegressionResult to initial state (NaN/zero values). nf is required to be set before calling this 
    function and must match the size of the arrays.

    Parameters:
    -----------
    lrr : LinearRegressionResult*
        Pointer to result to reset
    """
    lrr.calc_se = False
    lrr.calc_r2 = False
    lrr.status = 0
    lrr.df = 0
    lrr.r_squared = NAN
    lrr.r_squared_se = NAN
    memset(lrr.beta, 0, lrr.nf * sizeof(double))
    memset(lrr.beta_se, 0, lrr.nf * sizeof(double))


cdef inline LinearRegressionResult* lrr_array_new(size_t count, size_t nf) noexcept nogil:
    """
    Allocate and initialize an array of LinearRegressionResult structures.
    Returns NULL on failure.
    """
    if count == 0:
        return NULL

    cdef LinearRegressionResult* lrr_array = <LinearRegressionResult*> malloc(count * sizeof(LinearRegressionResult))
    if lrr_array == NULL:
        return NULL

    cdef size_t i
    cdef bint info

    for i in range(count):
        info = lrr_init(&lrr_array[i], nf)
        if info != 0:
            lrr_array_free(lrr_array, i + 1)
            return NULL

    return lrr_array


cdef inline void lrr_array_free(LinearRegressionResult* lrr_array, size_t count) noexcept nogil:
    """
    Free an array of LinearRegressionResult structures.
    Safe to call on partially initialized arrays or NULL pointer.
    """
    if lrr_array == NULL:
        return

    cdef size_t i
    for i in range(count):
        lrr_free(&lrr_array[i])

    free(lrr_array)

# =============================================================================
# Extension types (for testing purposes)
# =============================================================================

cdef class CyLinearRegression:
    """
    Fast linear regression accumulator using Cython and LAPACK.

    This class provides an efficient way to incrementally build linear regression
    models by accumulating observations and computing results on demand.

    Parameters
    ----------
    n_features : int
        Number of features (excluding intercept). Intercept is automatically included.

    Attributes
    ----------
    n_features : int
        Number of features excluding intercept
    n_params : int
        Number of parameters including intercept
    count : int
        Number of observations added
    """
    cdef LinearRegressionState* _state
    cdef readonly int n_features
    cdef readonly int n_params
    cdef bint _owns_state

    def __cinit__(self, n_features: int = 1):
        if n_features <= 0:
            raise ValueError("n_features must be positive")

        self.n_features = n_features
        self.n_params = n_features + 1  # +1 for intercept
        self._state = lrs_new(self.n_params)
        self._owns_state = True

        if self._state == NULL:
            raise MemoryError("Failed to allocate LinearRegressionState")

    def __dealloc__(self):
        if self._owns_state and self._state != NULL:
            lrs_free(self._state)

    @property
    def count(self):
        """Number of observations that have been added."""
        return self._state.count if self._state != NULL else 0

    def reset(self):
        """Reset the accumulator to initial state (removes all observations)."""
        if self._state != NULL:
            lrs_reset(self._state)

    def add(self, double y, cnp.ndarray[double, ndim=1] x):
        """
        Add a single observation to the regression.

        Parameters
        ----------
        y : float
            Target/response variable
        x : np.ndarray[float64, ndim=1]
            Feature vector of length n_features (excluding intercept)
        """
        if self._state == NULL:
            raise RuntimeError("LinearRegressionState not initialized")

        if x.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[0]}")

        cdef double[:] x_view = x
        lrs_add(self._state, y, x_view)

    def add_batch(self, cnp.ndarray[double, ndim=1] y, cnp.ndarray[double, ndim=2] X):
        """
        Add multiple observations at once.

        Parameters
        ----------
        y : np.ndarray[float64, ndim=1]
            Target values of shape (n_samples,)
        X : np.ndarray[float64, ndim=2]
            Feature matrix of shape (n_samples, n_features)
        """
        if self._state == NULL:
            raise RuntimeError("LinearRegressionState not initialized")

        if y.shape[0] != X.shape[0]:
            raise ValueError("y and X must have same number of samples")

        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        cdef:
            size_t i, n_samples = y.shape[0]
            double[:] y_view = y
            double[:, :] X_view = X
            double[:] x_row

        with nogil:
            for i in range(n_samples):
                x_row = X_view[i, :]
                lrs_add(self._state, y_view[i], x_row)

    def merge(self, CyLinearRegression other):
        """
        Merge another CyLinearRegression accumulator into this one.

        Parameters
        ----------
        other : CyLinearRegression
            Another regression accumulator with same n_features
        """
        if self._state == NULL or other._state == NULL:
            raise RuntimeError("LinearRegressionState not initialized")

        if self.n_features != other.n_features:
            raise ValueError("Cannot merge regressors with different n_features")

        lrs_merge(self._state, other._state)

    def compute(self, calc_se: bool = True, calc_r2: bool = True):
        """
        Compute regression results from accumulated observations.

        Returns
        -------
        CyRegressionResult
            Object containing coefficients, standard errors, R-squared, etc.
        """
        if self._state == NULL:
            raise RuntimeError("LinearRegressionState not initialized")

        cdef LinearRegressionResult* result = lrr_new(self.n_params)
        if result == NULL:
            raise MemoryError("Failed to allocate CyRegressionResult")

        try:
            lrs_to_result(self._state, result, calc_se, calc_r2)
            return CyRegressionResult._from_c_struct(result, self.n_params)
        finally:
            lrr_free(result)

    def copy(self):
        """Create a deep copy of this CyLinearRegression class."""
        cdef CyLinearRegression new_lr = CyLinearRegression(self.n_features)
        if self._state != NULL and new_lr._state != NULL:
            lrs_merge(new_lr._state, self._state)
        return new_lr


cdef class CyRegressionResult:
    """
    Results from a linear regression computation.

    Attributes
    ----------
    beta : np.ndarray
        Regression coefficients (intercept first, then features)
    beta_se : np.ndarray
        Standard errors of coefficients
    r_squared : float
        R-squared statistic
    df : int
        Degrees of freedom for residuals
    """
    def __cinit__(self):
        pass

    @staticmethod
    cdef CyRegressionResult _from_c_struct(LinearRegressionResult* result, int nf):
        """Create CyRegressionResult from C struct (internal use only)."""
        cdef CyRegressionResult py_result = CyRegressionResult()

        py_result.status = result.status
        py_result.calc_se = result.calc_se
        py_result.calc_r2 = result.calc_r2

        py_result.nf = nf
        py_result.r_squared = result.r_squared
        py_result.r_squared_se = result.r_squared_se
        py_result.df = int(result.df)

        # Copy coefficient arrays
        py_result.beta = np.empty(nf, dtype=np.float64)
        py_result.beta_se = np.empty(nf, dtype=np.float64)

        cdef:
            double[:] beta_view = py_result.beta
            double[:] beta_se_view = py_result.beta_se
            int i

        for i in range(nf):
            beta_view[i] = result.beta[i]
            beta_se_view[i] = result.beta_se[i]

        return py_result

    @property
    def ok(self):
        """Whether the regression computation was successful."""
        return self.status == 0

    @property
    def intercept(self):
        """Intercept term (first coefficient)."""
        return self.beta[0]

    @property
    def slope(self):
        """Slope coefficients (excluding intercept)."""
        return self.beta[1:]

    @property
    def intercept_se(self):
        """Standard error of intercept."""
        return self.beta_se[0]

    @property
    def slope_se(self):
        """Standard errors of slope coefficients."""
        return self.beta_se[1:]

    def summary(self):
        """Return a summary string of the regression results."""
        lines = [
            f"Linear Regression Results",
            f"=" * 40,
            f"Status: {self.status} -> {self.status == 0}",
            f"Calculation of standard errors: {self.calc_se}",
            f"Calculation of R-squared: {self.calc_r2}",
            f"R-squared: {self.r_squared:.6f}",
            f"Degrees of freedom: {self.df}",
            f"",
            f"Coefficients:",
            f"{'Parameter':<12} {'Coeff':<12} {'Std Err':<12}"
        ]

        lines.append(f"{'Intercept':<12} {self.intercept:<12.6f} {self.intercept_se:<12.6f}")

        for i in range(len(self.slope)):
            lines.append(f"{'x' + str(i+1):<12} {self.slope[i]:<12.6f} {self.slope_se[i]:<12.6f}")

        return "\n".join(lines)

    def __repr__(self):
        return f"RegressionResult(n_params={self.n_params}, r_squared={self.r_squared:.4f}, df={self.df})"
