from enum import Enum
from typing import Any, Optional
from packaging.version import Version as _Version


class PreRelease(Enum):
    ALPHA = 'a'
    BETA = 'b'
    RELEASE_CANDIDATE = "rc"
    UNKNOWN = "unknown"

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
    
    @classmethod
    def _missing_(cls, value):
        return PreRelease.UNKNOWN


class Version(_Version):
    """
    The version class

    - Properties:
        - main_version: An `int` of the main version
        - minor_version: An `int` of the minor version
        - pre_release: An optional `PreRelease` of the pre-release version
        - pre_release_version: An `int` of the pre-release version
        - sub_version: An `int` of the sub version
    """
    @property
    def main_version(self) -> int:
        return self.release[0]

    @property
    def minor_version(self) -> int:
        return self.release[1]

    @property
    def pre_release(self) -> Optional[PreRelease]:
        if self.pre is not None:
            return PreRelease(self.pre[0])
        else:
            return None

    @property
    def pre_release_version(self) -> int:
        if self.pre is not None:
            return int(self.pre[1])
        else:
            return 0

    @property
    def sub_version(self) -> int:
        return self.release[2]

    def __init__(self, v: Any, /) -> None:
        try:
            # convert to string
            version_str = str(v)
            super().__init__(version_str)
        except Exception as e:
            raise ValueError(f"The given version '{v}' is not in valid version format.") from e

    def __eq__(self, other: Any) -> bool:
        other = Version(str(other))
        return super().__eq__(other)

    def __ge__(self, other: Any) -> bool:
        return self == other or self > other

    def __gt__(self, other: Any) -> bool:
        # convert to version
        other = Version(str(other))
        return super().__gt__(other)

    def __le__(self, other: Any) -> bool:
        return self == other or self < other

    def __lt__(self, other: Any) -> bool:
        # convert to version
        other = Version(str(other))
        return super().__lt__(other)
