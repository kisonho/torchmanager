# import typing modules
from typing import Any, Callable, List, Optional, Union

# import required modules
import torch

class Metric:
    '''
    The basic metric class

    - Parameters:
        - result: The `torch.Tensor` of average metric results
    '''
    # properties
    __metric_fn: Optional[Callable[[Any, Any], torch.Tensor]]
    __results: List[torch.Tensor]

    @property
    def result(self) -> torch.Tensor:
        return torch.tensor(self.__results).mean()

    def __init__(self, metric_fn: Optional[Callable[[Any, Any], torch.Tensor]]=None) -> None:
        '''
        Constructor

        - Parameters:
            - metric_fn: An optional `Callable` metrics function that accepts `Any` kind of prediction input and target and returns a metric `torch.Tensor`. A `call` method must be overriden if this parameter is set as `None`.
        '''
        self.__results = []
        self.__metric_fn = metric_fn

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        m = self.call(input, target)
        self.__results.append(m)
        return m
    
    def reset(self) -> None:
        '''Reset the current results list'''
        self.__results.clear()

    def call(self, input: Any, target: Any) -> torch.Tensor:
        '''
        Forward the current result method
        
        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        '''
        if self.__metric_fn is not None:
            return self.__metric_fn(input, target)
        else: raise NotImplementedError("[Metric Error]: metric_fn is not given.")

class Accuracy(Metric):
    '''The traditional accuracy metric to compare two `torch.Tensor`'''
    def call(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return input.eq(target).to(torch.float32).mean()

class SparseCategoricalAccuracy(Accuracy):
    '''The accuracy metric for normal integer labels'''
    def call(self, input: torch.Tensor, target: Union[torch.Tensor, int]) -> torch.Tensor:
        '''
        calculate the accuracy
        
        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        '''
        input = input.argmax(dim=1)
        return super().call(input, torch.tensor(target))

class CategoricalAccuracy(SparseCategoricalAccuracy):
    '''The accuracy metric for categorical labels'''
    def call(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        calculate the accuracy
        
        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The onehot label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        '''
        target = target.argmax(dim=1)
        return super().call(input, target)