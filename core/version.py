from .view import warnings

CURRENT = "v1.1.0a8"

class VersionError(SystemError):
    def __init__(self, method_name: str, maximum_supported_version: str) -> None:
        super().__init__(f"`{method_name}` has been deprecated and removed from version {maximum_supported_version}.")

def deprecated(target_version: str, removing_version: str):
    '''
    Deprecated decorator function

    - Parameters:
        - version: A `str` of deprecation
    '''
    # define wrapping function
    def wrapping_fn(fn):
        if CURRENT >= removing_version: raise VersionError(fn.__name__, removing_version)
        elif CURRENT >= target_version: warnings.warn(f"{fn.__name__} has been deprecated from {target_version} and will be removed from {removing_version}", DeprecationWarning)
        else: warnings.warn(f"{fn} will be deprecated from {target_version} and removed from {removing_version}", PendingDeprecationWarning)
        return fn
    return wrapping_fn