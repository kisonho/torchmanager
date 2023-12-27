class VersionError(SystemError):
    def __init__(self, method_name: str, maximum_supported_version: str, current_version: str) -> None:
        super().__init__(f"`{method_name}` has been deprecated and removed from version {maximum_supported_version}. Current version: {current_version}.")
