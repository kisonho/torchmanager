class LossError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Fetch loss failed.")

class MetricError(RuntimeError):
    __name: str

    @property
    def name(self) -> str:
        return self.__name

    def __init__(self, name: str) -> None:
        super().__init__(f"Fetch metric '{name}' failed.")
        self.__name = name

class StopTraining(RuntimeError):
    '''
    A runtime error to stop training
    
    - Properties:
        - epoch: An `int` of the epoch index when training stopped
    '''
    epoch: int

    def __init__(self, epoch: int, *args: object) -> None:
        super().__init__(*args)
        self.epoch = epoch