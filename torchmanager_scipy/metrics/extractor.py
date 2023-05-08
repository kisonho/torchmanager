from scipy import linalg
from torchmanager.metrics import FeatureMetric
from torchmanager_core import torch, view, _raise
from torchmanager_core.typing import Module, Optional


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
        if not self.use_linalg:
            view.warnings.warn("The `use_linalg` flag is set to `False`. The matrix times and square root of sigma will be calculated element-wisely, which may result in different calculation results than the actual matrix square root.")
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
