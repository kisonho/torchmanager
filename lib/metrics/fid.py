from torchmanager_core import torch, view
from torchmanager_core.typing import Generic, List, Module, Optional

from .metric import Metric


class FID(Metric, Generic[Module]):
    """
    Fréchet Inception Distance (FID) metric

    * Extends: `.metric.Metric`
    * Generic class of `Module`

    - Parameters:
        - feature_extractor: A `Module` to extract the features
        - input_features: A `list` of current features for generated images that extracted in `torch.Tensor`
        - result: A `torch.Tensor` of the current FID result
        - return_when_forwarding: A `bool` flag of if returning results when calculating the metrics, a `torch.nan` will be returned if set to `False` during forwarding
        - target_features: A `list` of current features for real images that extracted in `torch.Tensor`
    """
    feature_extractor: Optional[Module]
    """A `Module` to extract the features"""
    input_features: List[torch.Tensor]
    """A `list` of current features for generated images that extracted in `torch.Tensor`"""
    return_when_forwarding: bool
    """A `bool` flag of if returning results when calculating the metrics, a `torch.nan` will be returned if set to `False` during forwarding"""
    target_features: List[torch.Tensor]
    """A `list` of current features for real images that extracted in `torch.Tensor`"""

    @property
    def result(self) -> torch.Tensor:
        """A `torch.Tensor` of the current FID result"""
        # concat input and target
        input = torch.cat(self.input_features)
        target = torch.cat(self.target_features)

        # calculate mean and covariance
        mu_real = target.mean(0)
        mu_gen = input.mean(0)
        sigma_real = target.cov() / (target.shape[0] - 1)
        sigma_gen = input.cov() / (input.shape[0] - 1)
        sigma = sigma_real @ sigma_gen
        diff = mu_real - mu_gen

        # square root of sigma
        try:
            from scipy import linalg
            covmean = linalg.sqrtm(sigma)
            assert not isinstance(covmean, tuple), "The square root of `sigma` should not contain errest number."
            sigma = torch.from_numpy(covmean.real).to(sigma.device)
        except ImportError:
            view.warnings.warn("The `scipy` package is not installed to calculate matrix square root. The matrix square root will be calculated as an element-wise square root, which may result in different calculation results than the actual matrix square root.")
            sigma = sigma.sqrt()

        # Calculate the squared Euclidean distance between the means
        return diff @ diff + torch.trace(sigma_real + sigma_gen - 2 * sigma)

    def __init__(self, feature_extractor: Optional[Module] = None, return_when_forwarding: bool = True, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - feature_extractor: An optional `Module` to extract features, a pre-trained InceptionV3 will be used if not given
            - return_when_forwarding: A `bool` flag of if returning results when calculating the metrics, a `torch.nan` will be returned if set to `False` during forwarding
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(target=target)
        self.feature_extractor = feature_extractor
        self.input_features = []
        self.return_when_forwarding = return_when_forwarding
        self.target_features = []

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_features = self.forward_features(input)
        target_features = self.forward_features(target)
        self.input_features += [input_features.cpu().detach()]
        self.target_features += [target_features.cpu().detach()]
        return self.result if self.return_when_forwarding else torch.tensor(torch.nan)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
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

    def reset(self) -> None:
        super().reset()
        self.input_features.clear()
        self.target_features.clear()
