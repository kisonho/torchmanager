from torchmanager_core import torch
from torchmanager_core.errors import WithoutScipy
from torchmanager_core.typing import Any, Callable, Generic, Module, Optional

from .metric import Metric

FID = WithoutScipy


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
