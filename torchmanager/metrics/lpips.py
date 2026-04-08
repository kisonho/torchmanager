from torchmanager_core import functional as F, torch, raise_error
from torchmanager_core.typing import Enum

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
                model = _load_vgg(feature_extractor)
            case LPIPSNetType.VGG16:
                model = _load_squeezenet(feature_extractor)

        # set requires grad
        for param in model.parameters():
            param.requires_grad = False
        return model


class _LPIPSModule(torch.nn.Module):
    slices: torch.nn.ModuleList

    def __init__(self, slices: torch.nn.ModuleList) -> None:
        super().__init__()
        self.slices = slices

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # initialize output
        y: list[torch.Tensor] = []

        # forward features
        for slice in self.slices:
            f = slice(x)
            y.append(f)
            x = f
        return y


class LPIPS(FeatureMetric[None, _LPIPSModule | None]):
    """The LPIPS metric"""
    def __init__(self, feature_extractor: torch.nn.Sequential | None = None, net_type: LPIPSNetType | None = None, *, target: str | None = None) -> None:
        # check extractor and net type
        if feature_extractor is None:
            assert net_type is None, raise_error(TypeError("The pretrained net type must be `None` if feature extractor is not given."))
            lpips_module = None
        else:
            assert net_type is not None, raise_error(TypeError("The pretrained net type must be given if feature extractor is given."))

            # load lpips extractor
            lpips_module = _LPIPSModule(net_type.load(feature_extractor))

        # initialize feature metric
        super().__init__(feature_extractor=lpips_module, target=target)

    @torch.no_grad()
    def forward(self, input: list[torch.Tensor], target: list[torch.Tensor]) -> torch.Tensor:
        # Compute LPIPS: per-layer channel-normalized L2, spatially averaged, summed over layers
        dists: list[torch.Tensor] = []
        for xf, yf in zip(input, target):
            # Channel-wise unit normalization (as in LPIPS)
            xfn = F.normalize(xf, p=2, dim=1)
            yfn = F.normalize(yf, p=2, dim=1)

            # Squared difference, mean over C/H/W -> per-sample distance for this layer
            dist_l = ((xfn - yfn) ** 2).mean(dim=(1, 2, 3))  # [N]
            dists.append(dist_l)

        # Sum over layers -> [N], then average over batch
        lpips = torch.stack(dists, dim=1).sum(dim=1)  # [N]
        return lpips.mean()
