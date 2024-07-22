import functools, warnings
from typing import Optional, TypeVar, Union

from .details import CURRENT
from .errors import VersionError
from .version import Version

VersionType = Union[str, Version]

def deprecated(target_version: VersionType, removing_version: Optional[VersionType], *, current_version: VersionType = CURRENT):
    '''
    Deprecated decorator function

    - Parameters:
        - target_version: `Any` type of version for the deprecation
        - removing_version: `Any` type of version for removing
    '''
    # format versions
    target_version = Version(target_version)
    removing_version = Version(removing_version) if removing_version is not None else None
    current_version = Version(current_version)

    # define wrapping function
    def wrapping_fn(fn):
        @functools.wraps(fn)
        def deprecated_fn(*args, **kwargs):
            # check version
            if removing_version is None and current_version < target_version:
                warnings.warn(f"{fn} will be deprecated from {target_version} and removed from a future version.", PendingDeprecationWarning)
            elif removing_version is None and current_version >= target_version:
                warnings.warn(f"{fn.__name__} has been deprecated from {target_version} and will be removed from a future version", DeprecationWarning)
            elif removing_version is not None and current_version < target_version:
                warnings.warn(f"{fn} will be deprecated from {target_version} and removed from {removing_version}", PendingDeprecationWarning)
            elif current_version >= target_version:
                warnings.warn(f"{fn.__name__} has been deprecated from {target_version} and will be removed from {removing_version}", DeprecationWarning)
            else:
                raise VersionError(fn.__name__, str(removing_version))
            return fn(*args, **kwargs)
        return deprecated_fn
    return wrapping_fn
