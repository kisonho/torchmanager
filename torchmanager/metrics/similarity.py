import torch.nn.functional as F
from torchmanager_core import torch
from torchmanager_core.typing import Callable, Optional

from .metric import Metric


class PSNR(Metric):
    """
    The Peak Signal-to-Noise Ratio metric

    - Properties:
        - denormalize_fn: An optional `Callable` function to denormalize the images
        - max_val: A `float` of the maximum value of the input
    """
    denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
    max_val: float

    def __init__(self, max_val: float = 1.0, *, denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Constructor

        - Parameters:
            - max_val: A `float` of the maximum value of the input
            - denormalize_fn: An optional `Callable` function to denormalize the images
        """
        super().__init__()
        self.denormalize_fn = denormalize_fn
        self.max_val = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # denormalize input and target
        if self.denormalize_fn is not None:
            input = self.denormalize_fn(input)
            target = self.denormalize_fn(target)

        # calculate psnr with denomalized input and target
        mse = F.mse_loss(input, target)
        return 10 * torch.log10(1 / mse)


class SSIM(Metric):
    """
    The Structural Similarity Index metric

    - Properties:
        - window: A random gaussian window
        - window_size: An `int` of the window size
    """
    window: torch.Tensor

    @property
    def window_size(self) -> int:
        return self.window.shape[-1]

    def __init__(self, channels: int, /, sigma: float = 1.5, window_size: int = 11):
        """
        Constructor

        - Parameters:
            - channels: An `int` of the image channel
            - sigma: A `float` of the gaussian standard diviation
            - window_size: An `int` of the window size
        """
        super(SSIM, self).__init__()
        gauss = torch.Tensor(
            [torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))) for x in range(window_size)]
        )
        window = gauss / gauss.sum()
        window = window.unsqueeze(1)
        window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
        window = window.expand(channels, 1, window_size, window_size).contiguous()
        self.window = torch.nn.Parameter(window, requires_grad=False)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate mean
        mu1 = F.conv2d(input, self.window, padding=self.window_size // 2, groups=input.shape[1])
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=target.shape[1])
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # calculate sigma
        sigma1_sq = F.conv2d(input * input, self.window, padding=self.window_size // 2, stride=1, groups=input.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, stride=1, groups=target.shape[1]) - mu2_sq
        sigma12 = F.conv2d(input * target, self.window, padding=self.window_size // 2, stride=1, groups=input.shape[1]) - mu1_mu2

        # calculate ssim
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()
