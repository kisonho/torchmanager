import torch, torchmanager, torchmanager_core
from torchmanager.losses.loss import loss_fn
from torchmanager_core import Version, API_VERSION
from unittest import TestCase


class Test0102(TestCase):
    def test_dice_metric(self) -> None:
        from torchmanager import metrics

        # initialize
        dice_score_fn = metrics.Dice()
        partial_dice_score_fn = metrics.PartialDice(2)
        y = torch.randn((4, 4, 224, 224))
        y_test = torch.randn_like(y).argmax(1)

        # test dice score
        dice = dice_score_fn(y, y_test)
        partial_dice = partial_dice_score_fn(y, y_test)
        self.assertGreaterEqual(float(dice), 0, f"Dice value must be non-negative, got {dice}.")
        self.assertLessEqual(float(dice), 1, f"Dice value must be less or equal to 1, got {dice}.")
        self.assertGreaterEqual(float(partial_dice), 0, f"Dice value must be non-negative, got {partial_dice}.")
        self.assertLessEqual(float(partial_dice), 1, f"Dice value must be less or equal to 1, got {partial_dice}.")

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

    def test_sliding_window(self) -> None:
        from torchmanager import data

        # initialize
        x = torch.randn((3, 256, 256))

        # sliding window
        x_slides = data.sliding_window(x, (64, 64), (32, 32))
        self.assertEqual(x_slides.shape, (49, 3, 64, 64))

    def test_version(self) -> None:
        self.assertGreaterEqual(API_VERSION, "1.2")

    def test_py_version(self) -> None:
        import platform

        py_version = Version(platform.python_version())
        self.assertGreaterEqual(py_version, "3.8")
