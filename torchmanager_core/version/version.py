from enum import Enum
from typing import Any, Optional


class PreRelease(Enum):
    ALPHA = 'a'
    BETA = 'b'
    RELEASE_CANDIDATE = "rc"

    def __gt__(self, pre_release: Any) -> bool:
        # convert pre release
        if not isinstance(pre_release, PreRelease):
            pre_release = PreRelease(pre_release)

        # check pre release version
        if self == pre_release:
            return False
        elif self == PreRelease.RELEASE_CANDIDATE:
            return True
        elif self == PreRelease.BETA and pre_release == PreRelease.ALPHA:
            return True
        else:
            return False

    def __lt__(self, pre_release: Any) -> bool:
        # convert pre release
        if not isinstance(pre_release, PreRelease):
            pre_release = PreRelease(pre_release)

        # check pre release version
        if self == pre_release:
            return False
        elif pre_release == PreRelease.RELEASE_CANDIDATE:
            return True
        elif pre_release == PreRelease.BETA and self == PreRelease.ALPHA:
            return True
        else:
            return False

    def __le__(self, other: Any) -> bool:
        return self == other or self < other

    def __ge__(self, other: Any) -> bool:
        return self == other or self > other


class Version:
    """
    The version class

    - Properties:
        - main_version: An `int` of the main version
        - minor_version: An `int` of the minor version
        - pre_release: An optional `PreRelease` of the pre-release version
        - pre_release_version: An `int` of the pre-release version
        - sub_version: An `int` of the sub version
    """
    main_version: int
    minor_version: int
    pre_release: Optional[PreRelease]
    pre_release_version: int
    sub_version: int

    def __init__(self, v: Any, /) -> None:
        try:
            # convert to string
            version_str = str(v)
            version_str = version_str.split('+')[0]

            # format version
            if version_str.startswith('v'):
                version_str = version_str[1:]

            # split pre-release version
            if 'a' in version_str:
                pre_release_parts = version_str.split('a')
                self.pre_release = PreRelease.ALPHA
            elif 'b' in version_str:
                pre_release_parts = version_str.split('b')
                self.pre_release = PreRelease.BETA
            elif 'rc' in version_str:
                pre_release_parts = version_str.split('rc')
                self.pre_release = PreRelease.RELEASE_CANDIDATE
            else:
                pre_release_parts = [version_str, ""]
                self.pre_release = None
            version_str = pre_release_parts[0]
            self.pre_release_version = int(pre_release_parts[1]) if pre_release_parts[1] != '' else 0

            # split version
            version_parts = version_str.split('.')
            self.main_version = int(version_parts[0])
            self.minor_version = int(version_parts[1]) if len(version_parts) > 1 else 0
            self.sub_version = int(version_parts[2]) if len(version_parts) > 2 else 0
        except Exception as e:
            raise ValueError(f"The given version '{v}' is not in valid version format.") from e

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Version):
            return self.main_version == other.main_version and self.minor_version == other.minor_version and self.sub_version == other.sub_version and self.pre_release == other.pre_release and self.pre_release_version == other.pre_release_version
        else:
            other = Version(str(other))
            return self.__eq__(other)

    def __ge__(self, other: Any) -> bool:
        return self == other or self > other

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

    def __repr__(self) -> str:
        version_str = f"v{self.main_version}.{self.minor_version}"
        if self.sub_version > 0:
            version_str += f".{self.sub_version}"
        if self.pre_release is not None:
            pre_release_version = self.pre_release_version if self.pre_release_version > 0 else ""
            version_str += f"{self.pre_release.value}{pre_release_version}"
        return version_str
