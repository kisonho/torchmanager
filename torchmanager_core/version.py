import functools

from .typing import Any, Optional
from .view import warnings


class Version:
    main_version: int
    minor_version: int
    pre_release: Optional[str]
    pre_release_version: int
    sub_version: int

    def __init__(self, v: Any, /) -> None:
        # convert to string
        version_str = str(v)

        # format version
        if version_str.startswith('v'):
            version_str = version_str[1:]

        # split pre-release version
        if 'a' in version_str:
            pre_release_parts = version_str.split('a')
            self.pre_release = 'a'
        elif 'b' in version_str:
            pre_release_parts = version_str.split('b')
            self.pre_release = 'b'
        elif 'rc' in version_str:
            pre_release_parts = version_str.split('rc')
            self.pre_release = 'rc'
        else:
            pre_release_parts = [version_str, "0"]
            self.pre_release = None
        version_str = pre_release_parts[0]
        self.pre_release_version = int(pre_release_parts[1]) if pre_release_parts[1] != '' else 1

        # split version
        version_parts = version_str.split('.')
        self.main_version = int(version_parts[0])
        self.minor_version = int(version_parts[1])
        self.sub_version = int(version_parts[2]) if len(version_parts) > 2 else 0

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
            return self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version == other.sub_version and self.pre_release == other.pre_release and self.pre_release_version == other.pre_release_version
        else:
            other = Version(str(other))
            return self.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        # convert to version
        if not isinstance(other, Version):
            other = Version(str(other))

        # check version
        if self.main_version < other.main_version:
            return True
        elif self.main_version == other.main_version and self.minor_version < other.minor_version:
                return True
        elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version < other.sub_version:
            return True
        elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version == other.sub_version and self.pre_release is not None and other.pre_release is not None:
            return self.pre_release < other.pre_release or (self.pre_release == other.pre_release and self.pre_release_version < other.pre_release_version)
        elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version == other.sub_version and self.pre_release is not None:
            return True
        return False

    def __gt__(self, other: Any) -> bool:
        # convert to version
        if not isinstance(other, Version):
            other = Version(str(other))

        # check version
        if self.main_version > other.main_version:
            return True
        elif self.main_version == other.main_version and self.minor_version > other.minor_version:
            return True
        elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version > other.sub_version:
            return True
        elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version == other.sub_version and self.pre_release is not None and other.pre_release is not None:
            return self.pre_release > other.pre_release or (self.pre_release == other.pre_release and self.pre_release_version > other.pre_release_version)
        elif self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version == other.sub_version and other.pre_release is not None:
            return True
        return False

    def __le__(self, other: Any) -> bool:
        return self == other or self < other

    def __ge__(self, other: Any) -> bool:
        return self == other or self > other


API = Version("v1.2")
CURRENT = Version("v1.2rc")
DESCRIPTION: str = "PyTorch Training Manager {CURRENT}"


class VersionError(SystemError):
    def __init__(self, method_name: str, maximum_supported_version: str) -> None:
        super().__init__(f"`{method_name}` has been deprecated and removed from version {maximum_supported_version}. Current version: {CURRENT}.")


def deprecated(target_version: Any, removing_version: Any):
    '''
    Deprecated decorator function

    - Parameters:
        - target_version: `Any` type of version for the deprecation
        - removing_version: `Any` type of version for removing
    '''
    # format versions
    target_version = Version(target_version)
    removing_version = Version(removing_version)

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
