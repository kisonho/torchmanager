class ConfigsFormatError(Exception):
    '''
    A format error within the configs

    - Properties:
        - cfg_type: The type of the configs
    '''
    cfg_type: type

    def __init__(self, cfg_type: type, /) -> None:
        super().__init__(f"Cannot format configs to type `{cfg_type}`.")
        self.cfg_type = cfg_type
