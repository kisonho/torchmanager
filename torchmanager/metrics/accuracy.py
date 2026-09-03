from torchmanager_core import torch, Version, raise_error
from torchmanager_core.protocols import Reduction

from .conf_mat import BinaryConfusionMatrix
from .metric import Metric

__all__ = ["Accuracy", "SparseCategoricalAccuracy", "CategoricalAccuracy", "Dice", "F1", "HD95", "MAE", "PartialDice", "Precision", "Recall"]


class Accuracy(Metric):
    """
    The traditional accuracy metric to compare two `torch.Tensor`

    * extends: `.metric.Metric`
    * implements: `torchmanager_core.protocols.VersionConvertible`

    - Properties:
        - reduction: A `torchmanager_core.protocols.Reduction` of reduction method
    """
    reduction: Reduction

    def __init__(self, *, reduction: Reduction = Reduction.MEAN, target: str | None = None) -> None:
        super().__init__(target=target)
        self.reduction = reduction

    def convert(self, from_version: Version) -> None:
        if from_version < Version("v1.1"):
            self.reduction = Reduction.MEAN

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.reduction == Reduction.MEAN:
            return input.eq(target).to(torch.float32).mean()
        elif self.reduction == Reduction.SUM:
            return input.eq(target).to(torch.float32).sum()
        else:
            return input.eq(target)


class SparseCategoricalAccuracy(Accuracy):
    """
    The accuracy metric for normal integer labels

    * extends: `Accuracy`

    - Properties:
        - dim: An `int` of the probability dim index for the input
    """
    __dim: int
    _dim: int
    """Deprecated dimension property"""

    @property
    def dim(self) -> int:
        """The dimension of the class in `int`"""
        return self.__dim

    @dim.setter
    def dim(self, value: int) -> None:
        assert value >= 0, raise_error(ValueError(f"Dim must be a non-negative number, got {value}."))
        self.__dim = value

    def __init__(self, dim: int = -1, *, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - dim: An `int` of the classification dimension
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        self.dim = dim

    def convert(self, from_version: Version) -> None:
        super().convert(from_version)
        if from_version < Version("v1.3"):
            self.dim = self._dim

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.argmax(dim=self._dim) if input.shape[self._dim] > 1 else input > 0
        return super().forward(input, target)


class CategoricalAccuracy(SparseCategoricalAccuracy):
    """
    The accuracy metric for categorical labels

    * extends: `SparseCategoricalAccuracy`
    """

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.argmax(dim=self.dim)
        return super().forward(input, target)


class Dice(Metric):
    """The dice score metrics"""
    __dim: int
    __num_classes: int
    _dim: int
    """Deprecated dimension property"""
    _eps: float

    @property
    def dim(self) -> int:
        """The dimension of the class in `int`"""
        return self.__dim

    @dim.setter
    def dim(self, value: int) -> None:
        assert value >= 0, raise_error(ValueError(f"Dim must be a non-negative number, got {value}."))
        self.__dim = value

    @property
    def num_classes(self) -> int:
        """The number of classes in `int`"""
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, value: int) -> None:
        assert value > 0, raise_error(ValueError(f"Num classes must be a positive number, got {value}."))
        self.__num_classes = value

    def __init__(self, num_classes: int, /, dim: int = 1, *, eps: float = 1e-7, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - num_classes: The number of classes in `int`
            - dim: The class channel dimmension index in `int`
            - eps: A `float` of the small number to avoid zero divide
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        self.dim = dim
        self.num_classes = num_classes
        self._eps = eps

    def convert(self, from_version: Version) -> None:
        if from_version < Version("v1.3"):
            self.dim = self._dim

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert target to one-hot encoding
        input_dims = [s-1 for s in range(self.dim+1, len(input.shape))]
        target_one_hot = torch.nn.functional.one_hot(target, input.shape[self.dim]).permute(0, len(input.shape)-1, *input_dims)

        # Argmax the input and convert to one-hot encoding
        input_argmax = input.argmax(dim=self.dim)
        input = torch.nn.functional.one_hot(input_argmax, input.shape[self.dim]).permute(0, len(input.shape)-1, *input_dims)

        # Flatten the tensors
        input_flat = input.view(input.shape[0], self.num_classes, -1)
        target_flat = target_one_hot.view(target_one_hot.shape[0], self.num_classes, -1)

        # Calculate intersection and union for each class
        intersection = (input_flat * target_flat).sum(dim=2)
        union = input_flat.sum(dim=2) + target_flat.sum(dim=2)

        # Compute Dice score for each class
        dice = (2. * intersection + self._eps) / (union + self._eps)

        # Average Dice score across all classes
        dice = dice.mean(dim=1)
        return dice.mean()


class F1(BinaryConfusionMatrix):
    """The F1 metrics"""

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # calculate precision and recall
        precision = tp / (tp + fp + self._eps)
        recall = tp / (tp + fn + self._eps)

        # calculate F1
        f1 = 2 * precision * recall / (precision + recall + self._eps)
        return f1.mean()


class HD95(Metric):
    """
    The 95th-percentile Hausdorff distance between two binary masks.

    HD95 measures the separation between the non-zero points in ``input`` and
    ``target``. For every point in either mask, the distance to its nearest
    point in the other mask is calculated. The returned value is the 95th
    percentile of these bidirectional nearest-neighbor distances, making the
    metric less sensitive to isolated outliers than the maximum Hausdorff
    distance.

    Coordinates are expressed in tensor-index units, so the result is measured
    in pixels/voxels unless the caller scales the coordinate system beforehand.
    Equal empty masks have distance zero; if only one mask is empty, the
    distance is infinite because no correspondence exists.

    * extends: `.metrics.Metric`
    """

    def __init__(self, *, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - target: An optional `str` of target name in `input` and `target`
              during direct calling
        """
        super().__init__(target=target)

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate HD95 between the non-zero points of two masks."""
        # Convert each mask directly to its non-zero coordinates.
        input_points = input.nonzero().to(dtype=torch.float32)
        target_points = target.nonzero().to(dtype=torch.float32)

        # Define the otherwise ambiguous empty-set cases without constructing an invalid, empty pairwise-distance matrix.
        if (input_points.numel() == 0 or target_points.numel() == 0) and input_points.numel() == target_points.numel():
            return torch.zeros((), device=input.device)
        elif input_points.numel() == 0 or target_points.numel() == 0:
            return torch.full((), float("inf"), device=input.device)

        # Compute the pairwise distances once, then obtain both directed nearest-neighbor distances by reducing across its two dimensions.
        distances = torch.cdist(input_points, target_points)
        directed_distances = torch.cat((distances.amin(dim=1), distances.amin(dim=0)))

        # Combining both directions makes the result symmetric.
        return torch.quantile(directed_distances, 0.95)


class MAE(Metric):
    """
    The Mean Absolute Error metric

    * extends: `.metrics.Metric`

    - Properties:
        - reduction: A `.loss.Reduction` of reduction method
    """
    reduction: Reduction

    def __init__(self, *, reduction: Reduction = Reduction.MEAN, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - reduction: A `.loss.Reduction` of reduction method
            - target: An optional `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        self.reduction = reduction

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate MAE
        error = input - target
        error = error.abs()

        # error reduction
        if self.reduction == Reduction.MEAN:
            return error.mean()
        elif self.reduction == Reduction.SUM:
            return error.sum()
        else:
            return error


class PartialDice(Dice):
    """
    The partial dice score for a specific class index

    - Properties:
        - class_idx: An `int` of the target class index
    """
    class_idx: int

    def __init__(self, c: int, /, dim: int = 1, *, eps: float = 1e-7, target:str | None = None):
        """
        Constructor

        - Parameters:
            - c: The target class index in `int`
            - dim: The class channel dimmension index in `int`
            - eps: A `float` of the small number to avoid zero divide
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(dim, eps=eps, target=target)
        self.class_idx = c

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.argmax(self.dim)
        input_mask = input == self.class_idx
        target_mask = target == self.class_idx
        intersection = input_mask * target_mask
        dice = (2 * intersection.sum() + self._eps) / (input_mask.sum() + target_mask.sum() + self._eps)
        return dice.mean()


class Precision(BinaryConfusionMatrix):
    """The Precision metrics"""

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        precision = tp / (tp + fp + self._eps)
        return precision.mean()


class Recall(BinaryConfusionMatrix):
    """The Recall metrics"""

    def forward_metric(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        recall = tp / (tp + fn + self._eps)
        return recall.mean()
