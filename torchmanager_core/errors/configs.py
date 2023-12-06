class ConfigsFormatError(Exception):
    cfg_type: type

    def __init__(self, cfg_type: type, /) -> None:
        super().__init__(f"Cannot format configs to type `{cfg_type}`.")
