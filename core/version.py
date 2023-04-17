import functools

from .typing import Any, Optional
from .view import warnings


class Version:
    main_version: int
    minor_version: int
    pre_release: Optional[str]
    sub_version: int

    def __init__(self, v: Any, /) -> None:
        # convert to string
        version_str = str(v)

        # format version
        if version_str.startswith('v'):
            version_str = version_str[1:]

        # split version
        version_parts = version_str.split('.')
        self.main_version = int(version_parts[0])
        self.minor_version = int(version_parts[1])
        self.pre_release = None

        # set sub  version
        if len(version_parts) > 2:
            # split alpha subversion parts
            sub_version_parts = version_parts[2].split('a')
            if len(sub_version_parts) > 1:
                self.sub_version = int(sub_version_parts[0])
                self.pre_release = 'a' + sub_version_parts[1]
            else:
                sub_version_parts = version_parts[2].split('b')
                if len(sub_version_parts) > 1:
                    self.sub_version = int(sub_version_parts[0])
                    self.pre_release = 'b' + sub_version_parts[1]
                else:
                    self.sub_version = int(version_parts[2])
        else:
            self.sub_version = 0

    def __repr__(self) -> str:
        version_str = f"v{self.main_version}"
        if self.minor_version > 0 or self.sub_version > 0:
            version_str += f".{self.minor_version}"
        if self.sub_version > 0:
            version_str += f".{self.sub_version}"
        if self.pre_release is not None:
            version_str += self.pre_release
        return version_str

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Version):
            return self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version == other.sub_version and self.pre_release == other.pre_release
        else:
            other = Version(str(other))
            return self.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Version):
            if self.main_version < other.main_version:
                return True
            elif self.main_version == other.main_version and self.minor_version < other.minor_version:
                    return True
            elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version < other.sub_version:
                return True
            elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version == other.sub_version and self.pre_release is not None and other.pre_release is not None:
                return self.sub_version < other.sub_version
            return False
        else:
            other = Version(str(other))
            return self.__lt__(other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Version):
            if self.main_version > other.main_version:
                return True
            elif self.main_version == other.main_version and self.minor_version > other.minor_version:
                return True
            elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version > other.sub_version:
                return True
            elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version == other.sub_version and self.pre_release is not None and other.pre_release is not None:
                return self.sub_version > other.sub_version
            return False
        else:
            other = Version(str(other))
            return self.__gt__(other)

    def __le__(self, other: Any) -> bool:
        return self == other or self < other

    def __ge__(self, other: Any) -> bool:
        return self == other or self > other


API = Version("v1.1")
CURRENT = Version("v1.1.2")
DESCRIPTION: str = "PyTorch Training Manager v1.1.2"


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
            if CURRENT >= removing_version:
                raise VersionError(fn.__name__, removing_version)
            elif CURRENT >= target_version:
                warnings.warn(f"{fn.__name__} has been deprecated from {target_version} and will be removed from {removing_version}", DeprecationWarning)
            else:
                warnings.warn(f"{fn} will be deprecated from {target_version} and removed from {removing_version}", PendingDeprecationWarning)
            return fn(*args, **kwargs)
        return deprecated_fn
    return wrapping_fn
