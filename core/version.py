import functools
from .view import warnings

API: str = "v1.2"
CURRENT: str = "v1.2a4"
DESCRIPTION: str = "PyTorch Training Manager v1.2 (Alpha 4)"

class VersionError(SystemError):
    def __init__(self, method_name: str, maximum_supported_version: str) -> None:
        super().__init__(f"`{method_name}` has been deprecated and removed from version {maximum_supported_version}. Current version: {CURRENT}.")

def deprecated(target_version: str, removing_version: str):
    '''
    Deprecated decorator function

    - Parameters:
        - target_version: A `str` of version for the deprecation
        - removing_version: A `str` of version for removing
    '''
    # define wrapping function
    def wrapping_fn(fn):
        @functools.wraps(fn)
        def deprecated_fn(*args, **kwargs):
            if CURRENT >= removing_version: raise VersionError(fn.__name__, removing_version)
            elif CURRENT >= target_version: warnings.warn(f"{fn.__name__} has been deprecated from {target_version} and will be removed from {removing_version}", DeprecationWarning)
            else: warnings.warn(f"{fn} will be deprecated from {target_version} and removed from {removing_version}", PendingDeprecationWarning)
            return fn(*args, **kwargs)
        return deprecated_fn
    return wrapping_fn