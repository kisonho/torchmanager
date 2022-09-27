from .view import warnings

CURRENT_VERSION = "v1.1.0a5"

def deprecated(version: str, removing_version: str):
    '''
    Deprecated decorator function

    - Parameters:
        - version: A `str` of deprecation
    '''
    # define wrapping function
    def wrapping_fn(fn):
        if CURRENT_VERSION >= version: warnings.warn(f"{fn} has been deprecated from {version} and will be removed from {removing_version}", DeprecationWarning)
        else: warnings.warn(f"{fn} will be deprecated from {version} and removed from {removing_version}", PendingDeprecationWarning)
        return fn
    return wrapping_fn