from torchmanager_core import Version, abc, torch, _raise

from .metric import Metric


class BinaryConfusionMetric(Metric, abc.ABC):
    """
    The binary confusion metrics that calculates TP, FP, and FN and forward further to calculate the final metric

    * extends: `.metric.Metric`
    * Abstract class

    - Methods to implement:
        - forward_metric: The main method that accepts TP, TN, FP, and FN as `torch.Tensor` and returns the final metric as `torch.Tensor`
    """
    _class_index: int
    _dim: int
    _eps: float

    def __init__(self, dim: int = 1, *, class_index: int = 1, eps: float=1e-7, target: str | None = None):
        """
        Constructor

        - Parameters:
            - dim: The class channel dimmension index in `int`
            - class_index: The target class index in `int`
            - eps: A `float` of the small number to avoid zero divide
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        self._class_index = class_index
        self._dim = dim
        self._eps = eps

    def convert(self, from_version: Version) -> None:
        return super().convert(from_version)

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # argmax input
        input = input.argmax(dim=self._dim) if input.shape[self._dim] > 1 else input > 0

        # mask input and target
        input = input == self._class_index
        target = target == self._class_index

        # calculate TP, FP, and FN
        tp, tn, fp, fn = self.forward_conf_met(input.float(), target.float())
        return self.forward_metric(tp, tn, fp, fn)

    def forward_conf_met(self, input: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tp = torch.sum(target * input, dim=0).mean()
        tn = ((1 - target) * (1 - input)).sum(dim=0).mean()
        fp = torch.sum((1 - target) * input, dim=0).mean()
        fn = torch.sum(target * (1 - input), dim=0).mean()
        return tp, tn, fp, fn

    @abc.abstractmethod
    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        """
        The main method to calculate the final metric from confusion metrics

        - Parameters:
            - tp: The true positive `torch.Tensor`
            - tn: The true negative `torch.Tensor`
            - fp: The false positive `torch.Tensor`
            - fn: The false negative `torch.Tensor`
        - Returns: The final metric `torch.Tensor`
        """
        return NotImplemented


class ConfusionMatrix(Metric):
    """
    The confusion matrix metric

    * Extends: `.metric.Metric`
    """
    __num_classes: int

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    @property
    def result(self) -> torch.Tensor:
        return torch.tensor(torch.nan)

    @property
    def results(self) -> torch.Tensor:
        if len(self._results) == 0:
            return torch.zeros((self.num_classes, self.num_classes))
        else:
            conf_mat = torch.cat(self._results, dim=0).sum(dim=0)
            return conf_mat

    def __init__(self, num_classes: int, /, *, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - num_classes: An `int` of the total number of classes
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        assert num_classes > 0, _raise(ValueError(f"The number of classes must be a positive number, got {num_classes}."))
        self.__num_classes = num_classes

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # initialize metrics
        conf_mat = torch.zeros((self.num_classes, self.num_classes), device=input.device)

        # add confusion metrics
        for y_pred, y_true in zip(input, target):
            y_pred = y_pred.argmax(0)
            conf_mat += self.forward_hist(y_pred.flatten(), y_true.flatten())

        # calculate final metric
        return conf_mat

    def forward_hist(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate historgram

        - Parameters:
            - input: The prediction `torch.Tensor`, or `y_pred`
            - target: The label `torch.Tensor`, or `y_true`
        - Returns: A `torch.Tensor` of historgram
        """
        mask = (target >= 0) & (target < self.num_classes)
        hist = torch.bincount(self.num_classes * target[mask].to(torch.int64) + input[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
