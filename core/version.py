from .view import warnings

CURRENT = "v1.1.0a7"

def deprecated(target_version: str, removing_version: str):
    '''
    Deprecated decorator function

    - Parameters:
        - version: A `str` of deprecation
    '''
    # define wrapping function
    def wrapping_fn(fn):
        if CURRENT >= target_version: warnings.warn(f"{fn.__name__} has been deprecated from {target_version} and will be removed from {removing_version}", DeprecationWarning)
        else: warnings.warn(f"{fn} will be deprecated from {target_version} and removed from {removing_version}", PendingDeprecationWarning)
        return fn
    return wrapping_fn