from unittest import TestCase


class Test0105(TestCase):
    def test_py_version(self) -> None:
        import platform
        major, minor, _ = platform.python_version_tuple()
        self.assertGreaterEqual(int(major), 3)
        self.assertGreaterEqual(int(minor), 11)

    def test_version(self) -> None:
        from torchmanager_core import API_VERSION
        self.assertGreaterEqual(API_VERSION, "1.5")

    def test_lpips(self) -> None:
        from torchmanager.metrics.lpips import LPIPS, LPIPSNetType
        from torchmanager_core import torch

        alex = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(12)])
        squeeze = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(13)])
        vgg = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(30)])

        self.assertEqual(len(LPIPSNetType.ALEX.load(alex)), 5)
        self.assertEqual(len(LPIPSNetType.SQUEEZE.load(squeeze)), 7)
        self.assertEqual(len(LPIPSNetType.VGG16.load(vgg)), 5)

        metric = LPIPS()
        input_features = [torch.ones((2, 3, 4, 4))]
        target_features = [-torch.ones((2, 3, 4, 4))]

        result = metric(input_features, target_features)

        self.assertTrue(torch.isclose(result, torch.tensor(4.0)))
