from torchmanager_core import abc, torch, view, _raise
from torchmanager_core.typing import Any, Callable, Generic, TypeVar
from torchmanager_core.version import Version

from .metric import Metric

try:
    from scipy import linalg  # type: ignore
except ImportError:
    linalg = NotImplemented


M = TypeVar("M", bound=Callable[[Any, Any], torch.Tensor] | None)
Module = TypeVar("Module", bound=torch.nn.Module | None)


class FeatureMetric(Metric[M], Generic[M, Module]):
    """
    A metric that extracts inputs and targets with feature extractor and evaluates the extracted features instead of raw inputs

    * Extends: `.metric.Metric`
    * Generic class of `Module`

    - Parameters:
        - feature_extractor: A `Module` to extract the features
    """
    feature_extractor: Module
    """A `Module` to extract the features"""

    def __init__(self, metric_fn: M = None, feature_extractor: Module = None, *, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - feature_extractor: An optional `Module` to extract features, a pre-trained InceptionV3 will be used if not given
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(metric_fn, target=target)
        self.feature_extractor = feature_extractor

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        # unpack input and target
        input = input[self._target] if self._target is not None and isinstance(input, dict) else input
        target = target[self._target] if self._target is not None and isinstance(target, dict) else target

        # get features
        input_features = self.forward_features(input)
        target_features = self.forward_features(target)

        # rewrap input and target
        if self._target is not None:
            input_features, target_features = {self._target: input_features}, {self._target: target_features}
        return super().__call__(input_features, target_features)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor, /) -> torch.Tensor:
        """
        Extract features

        - Parameters:
            - x: A `torch.Tensor` to extract
        - Returns: A `torch.Tensor` of features if feature extractor is given
        """
        # get features]
        if self.feature_extractor is not None:
            return self.feature_extractor(x)
        else:
            return x

    def train(self, mode: bool = True):
        self.training = mode
        return self


class AccumulativeFeatureMetric(FeatureMetric[M, Module], abc.ABC):
    """
    A feature metric that accumulates the features across batches.
    
    * Extends: `FeatureMetric`
    * Generic: `M` and `Module`

    - Properties:
        - accumulative: A `bool` flag of if the metric is accumulative
        - features_fake: A `torch.Tensor` or `None` storing fake features
        - features_real: A `torch.Tensor` or `None` storing real features
        - result: A `torch.Tensor` representing the final score
    """
    accumulative: bool
    features_fake: torch.Tensor | None
    features_real: torch.Tensor | None

    @property
    def result(self) -> torch.Tensor:
        """The final KID score"""
        if self.accumulative and self.results is not None:
            return self.results[-1]
        else:
            return super().result

    def __init__(self, metric_fn: M = None, feature_extractor: Module = None, *, accumulative: bool = False, target: str | None = None) -> None:
        super().__init__(metric_fn, feature_extractor, target=target)
        self.accumulative = accumulative
        self.features_fake = None
        self.features_real = None

    @abc.abstractmethod
    def compute_score(self) -> torch.Tensor:
        """
        Compute the final score from the accumulated features

        - Returns: A `torch.Tensor` representing the final score
        """
        ...

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.features_fake = torch.cat([self.features_fake, input], dim=0) if self.features_fake is not None and self.accumulative else input
        self.features_real = torch.cat([self.features_real, target], dim=0) if self.features_real is not None and self.accumulative else target
        return self.compute_score()

    def reset(self) -> None:
        # Reset accumulated features
        self.features_real = None
        self.features_fake = None
        super().reset()


class ExtractorScore(FeatureMetric[M, Module]):
    """
    A general feature score metric which can be used as `InceptionScore` by taking the `feature_extractor` as an InceptionV3 model

    * Extends: `FeatureExtractorMetric`
    * Generic class of `Module`
    """
    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = input.softmax(1).mean(0)
        scores = scores * (scores / scores.mean()).log2()
        scores = torch.exp(scores.sum())
        return scores


class FID(AccumulativeFeatureMetric[None, Module]):
    """
    Fréchet Inception Distance (FID) metric

    * Extends: `FeatureExtractorMetric`
    * Generic class of `Module`

    - Properties:
        - use_linalg: A `bool` flag of if use `scipy.linalg` package
    """
    use_linalg: bool
    """A `bool` flag of if use `scipy.linalg` package"""

    def __init__(self, feature_extractor: Module = None, *, accumulative: bool = False, target: str | None = None, use_linalg: bool = True) -> None:
        """
        Constructor

        - Parameters:
            - feature_extractor: An optional `Module` to extract features, a pre-trained InceptionV3 will be used if not given
            - accumulative: A `bool` flag of if the metric is accumulative
            - target: A `str` of target name in `input` and `target` during direct calling
            - use_linalg: A `bool` flag of if use `scipy.linalg` package
        """
        super().__init__(feature_extractor=feature_extractor, accumulative=accumulative, target=target)
        self.use_linalg = use_linalg

    def convert(self, from_version: Version) -> None:
        if from_version < Version("v1.4"):
            self.accumulative = False
            self.features_fake = None
            self.features_real = None
        super().convert(from_version)

    def compute_score(self) -> torch.Tensor:
        # check if features are accumulated
        if self.features_real is None or self.features_fake is None:
            raise ValueError("No features accumulated. Ensure forward has been called before computing the score.")

        # calculate mean and covariance
        mu_real = self.features_real.mean(0)
        mu_gen = self.features_fake.mean(0)
        sigma_real = self.features_real.cov() / (self.features_real.shape[0] - 1)
        sigma_gen = self.features_fake.cov() / (self.features_fake.shape[0] - 1)
        diff = mu_real - mu_gen

        # square root of sigma
        if linalg is NotImplemented or not self.use_linalg:
            view.warnings.warn("The `scipy` package is not installed to calculate matrix square root or `use_linalg` is set to `False`. The matrix times and square root of sigma will be calculated element-wisely, which may result in different calculation results than the actual matrix square root.")
            sigma = sigma_real * sigma_gen
            sigma = sigma.sqrt()
        else:
            sigma = sigma_real @ sigma_gen
            covmean = linalg.sqrtm(sigma.cpu().numpy())
            assert not isinstance(covmean, tuple), _raise(TypeError("The square root of `covmean` should not contain errest number."))
            sigma = torch.from_numpy(covmean.real).to(sigma.device)  # type: ignore  # TODO: fix typing

        # Calculate the squared Euclidean distance between the means
        return diff @ diff + torch.trace(sigma_real + sigma_gen - 2 * sigma)

    def reset(self) -> None:
        super().reset()


class KID(AccumulativeFeatureMetric[None, Module]):
    """
    Kernel Inception Distance (KID) metric

    * Extends: `AccumulativeFeatureMetric`
    * Generic class of `Module`

    - Properties:
        - c: A `float` of small positive constant for numerical stability in the polynomial kernel
        - degree: An `int` representing the degree of the polynomial kernel
        - features_real: A `torch.Tensor` or `None` storing real features
        - features_fake: A `torch.Tensor` or `None` storing fake features
    """
    c: float
    degree: int

    def __init__(self, feature_extractor: Module = None, *, accumulative: bool = False, c: float = 1.0, degree: int = 3, scale: float = 100, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - feature_extractor: An optional `Module` to extract features
            - accumulative: A `bool` flag of if the metric is accumulative
            - c: A small positive constant for numerical stability in the polynomial kernel
            - degree: The degree of the polynomial kernel
            - scale: A `float` to scale the final score
            - target: A `str` of target name in `input` and `target` during direct
        """
        super().__init__(feature_extractor=feature_extractor, accumulative=accumulative, target=target)
        self.c = c
        self.degree = degree
        self.scale = scale

    def compute_score(self) -> torch.Tensor:
        # check if features are accumulated
        if self.features_real is None or self.features_fake is None:
            raise ValueError("No features accumulated. Ensure forward has been called before computing the score.")

        # Compute kernel matrices
        k_xx = self.polynomial_kernel(self.features_fake, self.features_real).sum()
        k_yy = self.polynomial_kernel(self.features_real, self.features_real).sum()
        k_xy = self.polynomial_kernel(self.features_fake, self.features_real).sum()
        m = self.features_fake.shape[0]

        # Compute unbiased MMD²
        mmd2 = k_xx + k_yy
        mmd2 /= m * (m - 1)
        mmd2 -= 2 * k_xy / (m ** 2)
        return self.scale * mmd2

    def polynomial_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the polynomial kernel between two sets of features

        - Parameters:
            - x: A `torch.Tensor` representing the first set of features
            - y: A `torch.Tensor` representing the second set of features
        - Returns: A `torch.Tensor` representing the kernel matrix
        """
        return (x @ y.T / x.shape[1] + self.c) ** self.degree
