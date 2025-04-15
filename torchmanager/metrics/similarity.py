import torch.nn.functional as F
from torchmanager_core import torch, Version, _raise
from torchmanager_core.typing import Callable

from .metric import Metric

_DEFAULT_MS_SSIM_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]


class CosineSimilarity(Metric):
    """The Cosine Similarity metric"""
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(input, target, dim=1).mean()


class PSNR(Metric):
    """
    The Peak Signal-to-Noise Ratio metric

    - Properties:
        - denormalize_fn: An optional `Callable` function to denormalize the images
    """
    __max_value: float | None
    denormalize_fn: Callable[[torch.Tensor], torch.Tensor] | None

    @property
    def max_value(self) -> float | None:
        return self.__max_value if hasattr(self, "_PSNR__max_value") else 1

    @max_value.setter
    def max_value(self, value: float | None) -> None:
        assert value is None or value > 0, _raise(ValueError("Max value must be positive."))
        self.__max_value = value

    def __init__(self, *, denormalize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None, max_value: float | None = 1, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - denormalize_fn: An optional `Callable` function to denormalize the images
            - max_value: An optional `float` of the maximum value
            - target: An optional `str` of the target name
        """
        super().__init__(target=target)
        self.denormalize_fn = denormalize_fn
        self.max_value = max_value

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # denormalize input and target
        if self.denormalize_fn is not None:
            input = self.denormalize_fn(input)
            target = self.denormalize_fn(target)

        # calculate psnr with denomalized input and target
        mse = F.mse_loss(input, target)
        max_value = input.max() if self.max_value is None else self.max_value
        return 10 * torch.log10(max_value ** 2 / mse)


class SSIM(Metric):
    """
    The Structural Similarity Index metric

    - Properties:
        - window: A random gaussian window
        - window_size: An `int` of the window size
    """
    __pixel_range: float
    denormalize_fn: Callable[[torch.Tensor], torch.Tensor] | None
    window: torch.Tensor

    @property
    def pixel_range(self) -> float:
        return self.__pixel_range

    @pixel_range.setter
    def pixel_range(self, value: float) -> None:
        assert value > 0, _raise(ValueError("Pixel range must be greater than 0."))
        self.__pixel_range = value

    @property
    def window_size(self) -> int:
        return self.window.shape[-1]

    def __init__(self, channels: int, /, sigma: float = 1.5, window_size: int = 11, *, denormalize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None, pixel_range: float = 255, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - channels: An `int` of the image channel
            - sigma: A `float` of the gaussian standard diviation
            - window_size: An `int` of the window size
            - pixel_range: A `float` of the pixel range
        """
        super(SSIM, self).__init__(target=target)
        gauss = torch.Tensor(
            [torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))) for x in range(window_size)]
        )
        window = gauss / gauss.sum()
        window = window.unsqueeze(1)
        window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
        window = window.expand(channels, 1, window_size, window_size).contiguous()
        self.window = torch.nn.Parameter(window, requires_grad=False)
        self.denormalize_fn = denormalize_fn
        self.pixel_range = pixel_range

    def convert(self, from_version: Version) -> None:
        if from_version < "1.3":
            self.denormalize_fn = None
            self.pixel_range = 255

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # denormalize input and target
        if self.denormalize_fn is not None:
            input = self.denormalize_fn(input)
            target = self.denormalize_fn(target)

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
        C1 = (0.01 * self.pixel_range) ** 2
        C2 = (0.03 * self.pixel_range) ** 2
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = ssim.clip(-1, 1)
        return ssim.mean()


class MS_SSIM(SSIM):
    """
    The Multi-Scale Structural Similarity Index metric

    - Properties:
        - weights: A `list` of the weights for each scale
        - scales: A `int` of the number of scales
    """
    __weights: list[float]

    @property
    def weights(self) -> list[float]:
        return self.__weights

    @weights.setter
    def weights(self, value: list[float]) -> None:
        assert len(value) > 0, _raise(ValueError("Weights must be a non-empty list."))
        self.__weights = value

    @property
    def scales(self) -> int:
        return len(self.weights)

    def __init__(self, channels: int, /, sigma: float = 1.5, window_size: int = 11, *, denormalize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None, pixel_range: float = 255, target: str | None = None, weights: list[float] = _DEFAULT_MS_SSIM_WEIGHTS) -> None:
        super().__init__(channels, sigma, window_size, denormalize_fn=denormalize_fn, pixel_range=pixel_range, target=target)
        self.weights = weights

    @torch.no_grad()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # initialize ms_ssim
        ms_ssims: list[torch.Tensor] = []

        # loop for scales
        for _ in self.weights:
            # calculate ssim
            ssim = super().forward(input, target)
            ms_ssims.append(ssim)

            # downsample input and target
            input = F.avg_pool2d(input, kernel_size=2, stride=2)
            target = F.avg_pool2d(target, kernel_size=2, stride=2)

        # calculate ms_ssim
        ms_ssim = torch.stack(ms_ssims)
        ms_ssim = ms_ssim * torch.tensor(self.weights).to(ms_ssim.device)
        return ms_ssim.mean()
