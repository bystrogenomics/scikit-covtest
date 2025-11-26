from sklearn.utils import check_array

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
    return check_array(X, ensure_2d=True, allow_nd=False, force_all_finite=True, dtype="numeric")
