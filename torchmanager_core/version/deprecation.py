import functools, warnings
from typing import Any

from .details import CURRENT
from .errors import VersionError
from .version import Version

def deprecated(target_version: Any, removing_version: Any, *, current_version: Any = CURRENT):
    '''
    Deprecated decorator function

    - Parameters:
        - target_version: `Any` type of version for the deprecation
        - removing_version: `Any` type of version for removing
    '''
    # format versions
    target_version = Version(target_version)
    removing_version = Version(removing_version)
    current_version = Version(current_version)

    # define wrapping function
    def wrapping_fn(fn):
        @functools.wraps(fn)
        def deprecated_fn(*args, **kwargs):
            if current_version >= removing_version:
                raise VersionError(fn.__name__, removing_version)
            elif current_version >= target_version:
                warnings.warn(f"{fn.__name__} has been deprecated from {target_version} and will be removed from {removing_version}", DeprecationWarning)
            else:
                warnings.warn(f"{fn} will be deprecated from {target_version} and removed from {removing_version}", PendingDeprecationWarning)
            return fn(*args, **kwargs)
        return deprecated_fn
    return wrapping_fn
