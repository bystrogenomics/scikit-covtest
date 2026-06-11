import numpy as np

try:
    import scipy.sparse as sp
except ImportError:  # sparse support optional
    sp = None


class InputValidationError(ValueError):
    """Exception raised for invalid input data."""


def _estimator_name(estimator):
    if estimator is None:
        return "This estimator"
    if isinstance(estimator, str):
        return estimator
    return estimator.__class__.__name__


def _asarray_with_order(x, dtype=None, order=None, copy=False):
    """Convert to numpy array with given dtype and memory order."""
    if order not in (None, "C", "F"):
        raise ValueError("order must be one of None, 'C', or 'F'.")

    arr = np.array(x, dtype=dtype, order=order, copy=copy)
    return arr


def _is_arraylike(x):
    return hasattr(x, "__array__") or hasattr(x, "__iter__")


def _num_samples(x):
    try:
        return len(x)
    except TypeError:
        return 0


def check_array(
    array,
    accept_sparse=False,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
    input_name="X",
):
    """
    Validate and convert input data to a compatible array or sparse matrix.

    This function is loosely modeled after sklearn.utils.check_array. It
    performs common validation steps such as:

    - converting array-like input to a numpy array or sparse matrix
    - checking dimensionality (1d, 2d, nd)
    - enforcing minimum numbers of samples and features
    - checking for NaN or infinite values
    - converting data type to numeric or a given dtype

    Parameters
    ----------
    array : array-like, sparse matrix, or similar
        Input data to be validated.

    accept_sparse : bool, str, or list of str, default False
        Whether to accept sparse matrices. If False, sparse inputs raise
        an error. If True, any scipy sparse format is accepted. If a
        string or list of strings, only specified formats are allowed
        (for example, "csr", "csc", "coo").

    accept_large_sparse : bool, default True
        If False, disallow sparse matrices with index dtype larger than
        int32 (for example int64 indices). Only relevant if accept_sparse
        is not False.

    dtype : str or numpy dtype, default "numeric"
        Desired data type of the output:

        - "numeric": convert to float64
        - any numpy dtype: convert to that dtype

    order : {"C", "F", None}, default None
        Desired memory layout of the output array. Only affects dense
        arrays. None means no specific order is enforced.

    copy : bool, default False
        If True, always return a new array or sparse matrix. If False,
        the input may be returned as is when possible.

    force_all_finite : bool or {"allow-nan"}, default True
        Whether to raise an error if the data contain NaN or infinite
        values.

        - True: do not allow NaN or infinite values
        - False: allow NaN and infinite values
        - "allow-nan": allow NaN but still forbid infinite values

    ensure_2d : bool, default True
        If True, raise an error if the input is not at least 2d. If the
        input is 1d and ensure_2d is True, it will be reshaped to
        (n_samples, 1).

    allow_nd : bool, default False
        If False, raise an error for arrays with more than 2 dimensions.

    ensure_min_samples : int, default 1
        Minimum number of samples (rows) required. Only checked when the
        input is at least 1d.

    ensure_min_features : int, default 1
        Minimum number of features (columns) required for 2d input.

    estimator : object or str, default None
        Estimator object or name used only to craft error messages.

    input_name : str, default "X"
        Name of the input used in error messages.

    Returns
    -------
    array_out : numpy.ndarray or scipy.sparse matrix
        Validated and converted input.

    Raises
    ------
    InputValidationError
        If validation checks fail.
    """
    est_name = _estimator_name(estimator)

    # Handle sparse separately if scipy is available
    is_sparse = sp is not None and sp.issparse(array)

    # 1. Sparse handling
    if is_sparse:
        if not accept_sparse:
            raise InputValidationError(
                f"{est_name} does not accept sparse input for {input_name}. "
                "Set accept_sparse=True or convert the data to dense."
            )

        # Normalize accept_sparse specification
        if isinstance(accept_sparse, bool):
            sparse_formats = None  # accept any
        elif isinstance(accept_sparse, str):
            sparse_formats = [accept_sparse]
        else:
            sparse_formats = list(accept_sparse)

        if sparse_formats is not None:
            fmt = array.getformat()
            if fmt not in sparse_formats:
                # convert to the first requested format
                array = array.asformat(sparse_formats[0])

        if not accept_large_sparse:
            ind_dtype = array.indices.dtype
            if ind_dtype not in (np.int32, np.int16, np.int8):
                raise InputValidationError(
                    f"{est_name} does not accept large sparse indices for {input_name}. "
                    "Set accept_large_sparse=True or convert to dense."
                )

        # dtype conversion for sparse
        if dtype == "numeric":
            desired_dtype = np.float64
        else:
            desired_dtype = np.dtype(dtype)

        if array.dtype != desired_dtype:
            array = array.astype(desired_dtype)

        # basic ndim check
        if array.ndim != 2:
            if ensure_2d:
                raise InputValidationError(
                    f"{est_name} expected 2d sparse input for {input_name}, "
                    f"got {array.ndim}d."
                )
            if not allow_nd and array.ndim > 2:
                raise InputValidationError(
                    f"{est_name} does not support {array.ndim}d sparse input "
                    f"for {input_name} when allow_nd=False."
                )

        # min samples / features checks
        n_samples, n_features = array.shape
        if ensure_min_samples > 0 and n_samples < ensure_min_samples:
            raise InputValidationError(
                f"{est_name} requires at least {ensure_min_samples} samples "
                f"for {input_name}, but got {n_samples}."
            )
        if ensure_min_features > 0 and n_features < ensure_min_features:
            raise InputValidationError(
                f"{est_name} requires at least {ensure_min_features} features "
                f"for {input_name}, but got {n_features}."
            )

        # finite checks on sparse data operate on .data
        data = array.data
        if force_all_finite:
            if force_all_finite == "allow-nan":
                if not np.all(np.isfinite(data) | np.isnan(data)):
                    raise InputValidationError(
                        f"{input_name} contains infinite values, but "
                        "force_all_finite='allow-nan'."
                    )
            else:
                if not np.all(np.isfinite(data)):
                    raise InputValidationError(
                        f"{input_name} contains NaN or infinite values, but "
                        "force_all_finite=True."
                    )

        return array

    # 2. Dense or array-like handling
    # If it quacks like array: convert; else raise.
    if not _is_arraylike(array):
        raise InputValidationError(
            f"{est_name} expected array-like input for {input_name}, "
            f"got {type(array).__name__}."
        )

    # Convert to numpy array
    if dtype == "numeric":
        desired_dtype = np.float64
    else:
        desired_dtype = np.dtype(dtype)

    array = _asarray_with_order(
        array, dtype=desired_dtype, order=order, copy=copy
    )

    # Reject object dtype if numeric is requested but conversion failed
    if dtype == "numeric" and array.dtype.kind == "O":
        raise InputValidationError(
            f"{input_name} could not be converted to numeric dtype. "
            "Ensure it contains only numeric values."
        )

    # Dimensionality checks
    if array.ndim == 0:
        raise InputValidationError(
            f"{est_name} expected array-like input for {input_name}, "
            "got scalar."
        )

    if array.ndim == 1:
        if ensure_2d:
            # reshape (n_samples,) to (n_samples, 1)
            array = array.reshape(-1, 1)
    elif array.ndim > 2 and not allow_nd:
        raise InputValidationError(
            f"{est_name} does not support {array.ndim}d input for {input_name} "
            "when allow_nd=False."
        )

    # Min samples / features checks
    if array.ndim >= 1:
        n_samples = array.shape[0]
        if ensure_min_samples > 0 and n_samples < ensure_min_samples:
            raise InputValidationError(
                f"{est_name} requires at least {ensure_min_samples} samples "
                f"for {input_name}, but got {n_samples}."
            )

    if array.ndim >= 2:
        n_features = array.shape[1]
        if ensure_min_features > 0 and n_features < ensure_min_features:
            raise InputValidationError(
                f"{est_name} requires at least {ensure_min_features} features "
                f"for {input_name}, but got {n_features}."
            )

    # Finite checks for dense arrays
    if force_all_finite:
        if force_all_finite == "allow-nan":
            # allow NaN but no infinite values
            if not np.all(np.isfinite(array) | np.isnan(array)):
                raise InputValidationError(
                    f"{input_name} contains infinite values, but "
                    "force_all_finite='allow-nan'."
                )
        else:
            if not np.all(np.isfinite(array)):
                raise InputValidationError(
                    f"{input_name} contains NaN or infinite values, but "
                    "force_all_finite=True."
                )

    return array


def validate_data_matrix(X):
    """
    Validate that X is a 2D array of numeric values.

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    X_validated : ndarray
        Validated 2D array.
    """
    return check_array(
        X,
        ensure_2d=True,
        allow_nd=False,
        force_all_finite=True,
        dtype="numeric",
    )
