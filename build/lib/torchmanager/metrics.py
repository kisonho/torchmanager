# import typing modules
from typing import Any, Callable, List, Optional

# import required modules
import torch, warnings

class Metric(torch.nn.Module):
    '''
    The basic metric class

    * Could be use as a decorator of a function
    * Metric tensor is released from memory as soon as the result returned

    - Parameters:
        - result: The `torch.Tensor` of average metric results
    '''
    # properties
    __metric_fn: Optional[Callable[[Any, Any], torch.Tensor]]
    _results: List[torch.Tensor]

    @property
    def result(self) -> torch.Tensor:
        return torch.tensor(self._results).mean()

    def __init__(self, metric_fn: Optional[Callable[[Any, Any], torch.Tensor]]=None) -> None:
        '''
        Constructor

        - Parameters:
            - metric_fn: An optional `Callable` metrics function that accepts `Any` kind of prediction input and target and returns a metric `torch.Tensor`. A `call` method must be overriden if this parameter is set as `None`.
        '''
        super().__init__()
        self._results = []
        self.__metric_fn = metric_fn

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        m: torch.Tensor = super().__call__(input, target)
        self._results.append(m.detach())
        return m

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
    
    def reset(self) -> None:
        '''Reset the current results list'''
        self._results.clear()

    def forward(self, input: Any, target: Any) -> torch.Tensor:
        warnings.warn("[Pending Deprecation Warning]: The call method will be deprecated from v1.1.0, override the forward method instead.", PendingDeprecationWarning)
        return self.call(input, target)

class Accuracy(Metric):
    '''The traditional accuracy metric to compare two `torch.Tensor`'''
    def call(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return input.eq(target).to(torch.float32).mean()

class SparseCategoricalAccuracy(Accuracy):
    '''The accuracy metric for normal integer labels'''
    def call(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        calculate the accuracy
        
        - Parameters:
            - input: The prediction, or `y_pred`, in `Any` kind
            - target: The label, or `y_true`, in `Any` kind
        - Returns: The metric in `torch.Tensor`
        '''
        input = input.argmax(dim=-1)
        return super().call(input, target)

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