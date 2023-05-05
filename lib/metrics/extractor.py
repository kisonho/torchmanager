from torchmanager_core import torch, view, _raise
from torchmanager_core.typing import Any, Callable, Generic, Module, Optional

from .metric import Metric

try:
    from scipy import linalg  # type: ignore
except ImportError:
    linalg = NotImplemented


class FeatureMetric(Metric, Generic[Module]):
    """
    A metric that extracts inputs and targets with feature extractor and evaluates the extracted features instead of raw inputs

    * Extends: `.metric.Metric`
    * Generic class of `Module`

    - Parameters:
        - feature_extractor: A `Module` to extract the features
    """
    feature_extractor: Optional[Module]
    """A `Module` to extract the features"""

    def __init__(self, metric_fn: Optional[Callable[[Any, Any], torch.Tensor]] = None, feature_extractor: Optional[Module] = None, *, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - feature_extractor: An optional `Module` to extract features, a pre-trained InceptionV3 will be used if not given
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(metric_fn, target=target)
        self.feature_extractor = feature_extractor

    def __call__(self, input: Any, target: Any) -> torch.Tensor:
        input_features = self.forward_features(input)
        target_features = self.forward_features(target)
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


class ExtractorScore(FeatureMetric[Module]):
    """
    A general feature score metric which can be used as `InceptionScore` by taking the `feature_extractor` as an InceptionV3 model

    * Extends: `FeatureExtractorMetric`
    * Generic class of `Module`
    * When forwarding this metric, `target` (real images) parameter is not required
    """

    def __call__(self, input: Any, target: Any = None) -> torch.Tensor:
        input_features = self.forward_features(input)
        return super().__call__(input_features, target)

    def forward(self, input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        scores = input.softmax(1).mean(0)
        scores = scores * (scores / scores.mean()).log2()
        scores = torch.exp(scores.sum())
        return scores


class FID(FeatureMetric[Module]):
    """
    Fréchet Inception Distance (FID) metric

    * Extends: `FeatureExtractorMetric`
    * Generic class of `Module`
    """
    use_linalg: bool
    """use_linalg: A `bool` flag of if use `scipy.linalg` package"""

    def __init__(self, feature_extractor: Optional[Module] = None, *, use_linalg: bool = True, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - feature_extractor: An optional `Module` to extract features, a pre-trained InceptionV3 will be used if not given
            - return_when_forwarding: A `bool` flag of if returning results when calculating the metrics, a `torch.nan` will be returned if set to `False` during forwarding
            - target: A `str` of target name in `input` and `target` during direct calling
            - use_linalg: A `bool` flag of if use `scipy.linalg` package
        """
        super().__init__(feature_extractor=feature_extractor, target=target)
        self.use_linalg = use_linalg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate mean and covariance
        mu_real = target.mean(0)
        mu_gen = input.mean(0)
        sigma_real = target.cov() / (target.shape[0] - 1)
        sigma_gen = input.cov() / (input.shape[0] - 1)
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
            sigma = torch.from_numpy(covmean.real).to(sigma.device)

        # Calculate the squared Euclidean distance between the means
        return diff @ diff + torch.trace(sigma_real + sigma_gen - 2 * sigma)

    def reset(self) -> None:
        super().reset()
