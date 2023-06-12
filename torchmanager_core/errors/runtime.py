class PredictionError(RuntimeError):
    '''A runtime error that prediction stopped unexpectedly'''
    def __init__(self) -> None:
        super().__init__("Prediction failed.")


class TestingError(RuntimeError):
    '''A runtime error that testing stopped unexpectedly'''
    def __init__(self) -> None:
        super().__init__("Testing failed.")
