import torch, torchmanager, torchmanager_core
from torchmanager_core import Version, API_VERSION
from unittest import TestCase


class Test0102(TestCase):
    def test_random(self) -> None:
        from torchmanager_core.random import freeze_seed, unfreeze_seed

        seed = 1
        freeze_seed(1)
        random_num_1 = torch.randint(0, 1, [1])
        freeze_seed(1)
        random_num_2 = torch.randint(0, 1, [1])
        self.assertEqual(int(random_num_1), int(random_num_2))

        random_seed = unfreeze_seed()
        self.assertNotEqual(seed, random_seed, f"Seed not unfrozon: current_seed={seed}")

    def test_version(self) -> None:
        self.assertGreaterEqual(API_VERSION, "1.2")

    def test_py_version(self) -> None:
        import platform

        py_version = Version(platform.python_version())
        self.assertGreaterEqual(py_version, "3.8")
