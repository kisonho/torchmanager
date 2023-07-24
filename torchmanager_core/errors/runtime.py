class PredictionError(RuntimeError):
    '''A runtime error that prediction stopped unexpectedly'''
    def __init__(self) -> None:
        super().__init__("Prediction failed.")


class TestingError(RuntimeError):
    '''A runtime error that testing stopped unexpectedly'''
    def __init__(self) -> None:
        super().__init__("Testing failed.")


class TransformError(RuntimeError):
    def __init__(self, transfrom: object, data: object) -> None:
        super().__init__(f"Transform function {transfrom} failed: data={data}.")
