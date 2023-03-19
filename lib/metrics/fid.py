from torchmanager_core import torch
from torchmanager_core.typing import List, Optional, Self
from torchvision.models import inception_v3

from .metric import Metric


class FID(Metric):
    """
    Fréchet Inception Distance (FID) metric

    - Parameters:
        - feature_extractor: A `torch.nn.Module` to extract the features
        - input_features: A `list` of current features for generated images that extracted in `torch.Tensor`
        - result: A `torch.Tensor` of the current FID result
        - return_when_forwarding: A `bool` flag of if returning results when calculating the metrics, a `torch.nan` will be returned if set to `False` during forwarding
        - target_features: A `list` of current features for real images that extracted in `torch.Tensor`
    """
    feature_extractor: torch.nn.Module
    """A `torch.nn.Module` to extract the features"""
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
        sigma_real = target.cov()
        sigma_gen = input.cov()

        # Calculate the squared Euclidean distance between the means
        diff = (mu_real - mu_gen) ** 2
        sigma = torch.trace(sigma_real) + torch.trace(sigma_gen) - 2 * torch.trace((torch.dot(sigma_real, sigma_gen).sqrt()))
        fid_score = diff + sigma
        return fid_score

    def __init__(self, feature_extractor: Optional[torch.nn.Module] = None, return_when_forwarding: bool = True, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - feature_extractor: An optional `torch.nn.Module` to extract features, a pre-trained InceptionV3 will be used if not given
            - return_when_forwarding: A `bool` flag of if returning results when calculating the metrics, a `torch.nan` will be returned if set to `False` during forwarding
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        super(FID, self).__init__(target=target)
        self.return_when_forwarding = return_when_forwarding
        self.input_features = []
        self.target_features = []

        # initialize inception model
        if feature_extractor is None:
            self.feature_extractor = inception_v3(pretrained=True, transform_input=False)
            self.feature_extractor.fc = torch.nn.Identity()  # Remove last layer # type: ignore
            self.feature_extractor.eval()
        else:
            self.feature_extractor = feature_extractor

        # disable gradients
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_features = self.forward_features(input)
        target_features = self.forward_features(target)
        self.input_features.append(input_features)
        self.target_features.append(target_features)
        return self.result if self.return_when_forwarding else torch.tensor(torch.nan)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features

        - Parameters:
            - x: A `torch.Tensor` to extract
        - Returns: A `torch.Tensor` of features
        """
        # get features]
        features: torch.Tensor = self.feature_extractor(x)
        features = features.squeeze(3).squeeze(2)
        return features
    
    def train(self, mode: bool = True) -> Self:
        self.training = mode
        return self

    def reset(self) -> None:
        super().reset()
        self.input_features.clear()
        self.target_features.clear()