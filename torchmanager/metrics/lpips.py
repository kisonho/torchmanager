from torchmanager_core import functional as F, torch, raise_error
from torchmanager_core.typing import Enum, cast

from .extractor import FeatureMetric

__all__ = ["LPIPSNetType", "LPIPS"]


def _load_alexnet(model: torch.nn.Sequential) -> torch.nn.ModuleList:
    slice1 = torch.nn.Sequential()
    slice2 = torch.nn.Sequential()
    slice3 = torch.nn.Sequential()
    slice4 = torch.nn.Sequential()
    slice5 = torch.nn.Sequential()
    for x in range(2):
        slice1.add_module(str(x), model[x])
    for x in range(2, 5):
        slice2.add_module(str(x), model[x])
    for x in range(5, 8):
        slice3.add_module(str(x), model[x])
    for x in range(8, 10):
        slice4.add_module(str(x), model[x])
    for x in range(10, 12):
        slice5.add_module(str(x), model[x])
    return torch.nn.ModuleList([slice1, slice2, slice3, slice4, slice5])

def _load_squeezenet(model: torch.nn.Sequential) -> torch.nn.ModuleList:
    slice1 = torch.nn.Sequential()
    slice2 = torch.nn.Sequential()
    slice3 = torch.nn.Sequential()
    slice4 = torch.nn.Sequential()
    slice5 = torch.nn.Sequential()
    slice6 = torch.nn.Sequential()
    slice7 = torch.nn.Sequential()
    for x in range(2):
        slice1.add_module(str(x), model[x])
    for x in range(2,5):
        slice2.add_module(str(x), model[x])
    for x in range(5, 8):
        slice3.add_module(str(x), model[x])
    for x in range(8, 10):
        slice4.add_module(str(x), model[x])
    for x in range(10, 11):
        slice5.add_module(str(x), model[x])
    for x in range(11, 12):
        slice6.add_module(str(x), model[x])
    for x in range(12, 13):
        slice7.add_module(str(x), model[x])
    return torch.nn.ModuleList([slice1, slice2, slice3, slice4, slice5, slice6, slice7])

def _load_vgg(model: torch.nn.Sequential) -> torch.nn.ModuleList:
    slice1 = torch.nn.Sequential()
    slice2 = torch.nn.Sequential()
    slice3 = torch.nn.Sequential()
    slice4 = torch.nn.Sequential()
    slice5 = torch.nn.Sequential()
    for x in range(4):
        slice1.add_module(str(x), model[x])
    for x in range(4, 9):
        slice2.add_module(str(x), model[x])
    for x in range(9, 16):
        slice3.add_module(str(x), model[x])
    for x in range(16, 23):
        slice4.add_module(str(x), model[x])
    for x in range(23, 30):
        slice5.add_module(str(x), model[x])
    return torch.nn.ModuleList([slice1, slice2, slice3, slice4, slice5])


class LPIPSNetType(Enum):
    """The pre-trained LPIPS network types"""
    ALEX = 'alex'
    SQUEEZE = 'squeeze'
    VGG16 = 'vgg16'

    def load(self, feature_extractor: torch.nn.Sequential) -> torch.nn.ModuleList:
        """
        Load the network feature extractors

        Returns: A `torch.nn.ModuleList` of feature extractors
        """
        # load pretrained model
        match self:
            case LPIPSNetType.ALEX:
                model = _load_alexnet(feature_extractor)
            case LPIPSNetType.SQUEEZE:
                model = _load_squeezenet(feature_extractor)
            case LPIPSNetType.VGG16:
                model = _load_vgg(feature_extractor)

        # set requires grad
        for param in model.parameters():
            param.requires_grad = False
        return model


class _ScalingLayer(torch.nn.Module):
    shift: torch.Tensor
    scale: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188]).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor([.458, .448, .450]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) / self.scale


class _LPIPSModule(torch.nn.Module):
    scaling_layer: _ScalingLayer
    slices: torch.nn.ModuleList

    def __init__(self, slices: torch.nn.ModuleList) -> None:
        super().__init__()
        self.scaling_layer = _ScalingLayer()
        self.slices = slices

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # initialize output
        y: list[torch.Tensor] = []
        x = self.scaling_layer(x)

        # forward features
        for slice in self.slices:
            f = slice(x)
            y.append(f)
            x = f
        return y


class LPIPS(FeatureMetric[None, _LPIPSModule | None]):
    """The LPIPS metric"""
    lins: torch.nn.ModuleList | None

    def __init__(self, feature_extractor: torch.nn.Sequential | None = None, net_type: LPIPSNetType | None = None, lin_layers: torch.nn.ModuleList | None = None, *, target: str | None = None) -> None:
        # check extractor and net type
        if feature_extractor is None:
            assert net_type is None, raise_error(TypeError("The pretrained net type must be `None` if feature extractor is not given."))
            lpips_module = None
        else:
            assert net_type is not None, raise_error(TypeError("The pretrained net type must be given if feature extractor is given."))

            # load lpips extractor and learned calibration layers
            lpips_module = _LPIPSModule(net_type.load(feature_extractor))

        # initialize feature metric
        super().__init__(feature_extractor=lpips_module, target=target)
        self.lins = lin_layers

    @torch.no_grad()
    def forward(self, input: list[torch.Tensor], target: list[torch.Tensor]) -> torch.Tensor:
        # Compute LPIPS: per-layer channel-normalized L2, spatially averaged, summed over layers
        dists: list[torch.Tensor] = []
        for i, (xf, yf) in enumerate(zip(input, target)):
            # Channel-wise unit normalization (as in LPIPS)
            xfn = F.normalize(xf, p=2, dim=1)
            yfn = F.normalize(yf, p=2, dim=1)
            diff = (xfn - yfn) ** 2

            if self.lins is not None:
                # Learned LPIPS calibration: 1x1 conv then spatial average.
                dist_l = cast(torch.Tensor, self.lins[i](diff)).mean(dim=(1, 2, 3))
            else:
                # Baseline fallback when learned LPIPS weights are unavailable.
                dist_l = diff.sum(dim=1).mean(dim=(1, 2))
            dists.append(dist_l)

        # Sum over layers -> [N], then average over batch
        lpips = torch.stack(dists, dim=1).sum(dim=1)
        return lpips.mean()
