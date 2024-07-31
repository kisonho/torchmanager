import torch, torchmanager, torchmanager_core
from torchmanager.losses.loss import loss_fn
from torchmanager_core import Version, API_VERSION
from unittest import TestCase


class Test0102(TestCase):
    def test_dice_metric(self) -> None:
        from torchmanager import metrics

        # initialize
        dice_score_fn = metrics.Dice(4)
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

    def test_f1_score(self) -> None:
        from torchmanager import metrics

        # initialize
        f1_score_fn = metrics.F1()
        y = torch.randn((1, 1, 64, 64))
        y_test = torch.randn_like(y).argmax(1)

        # test f1 score
        f1 = f1_score_fn(y, y_test)
        self.assertGreaterEqual(float(f1_score_fn.result), 0, f"F1 Score value must be non-negative, got {f1}.")

        '''
        # Test with sklearn.metrics.f1_score
        from sklearn.metrics import f1_score
        y = y.argmax(1) == 1
        y_test = y_test == 1
        f1_sklearn = f1_score(y_test.flatten(), y.flatten(), average='binary')
        self.assertAlmostEqual(float(f1_score_fn.result), f1_sklearn, places=2)
        '''

    def test_miou(self) -> None:
        from torchmanager import metrics

        # initialize
        miou_score_fn = metrics.MeanIoU()
        y = torch.randn((1, 1, 64, 64))
        y_test = torch.randn_like(y) > 0

        # test miou score
        miou = miou_score_fn(y, y_test)
        self.assertGreaterEqual(float(miou_score_fn.result), 0, f"Mean IoU value must be non-negative, got {miou}.")

        '''
        # Test with sklearn.metrics.jaccard_score
        from sklearn.metrics import jaccard_score
        miou_jaccard = jaccard_score(y_test.flatten(), (y > 0).flatten(), average='weighted')
        self.assertAlmostEqual(float(miou_score_fn.result), miou_jaccard, places=2)
        '''

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

    def test_ssim(self) -> None:
        from torchmanager.metrics import SSIM

        # initialize
        x = torch.randn((1, 3, 256, 256))
        y = torch.randn_like(x)
        ssim_fn = SSIM(3)

        # calculate ssim
        ssim = ssim_fn(x, y)
        self.assertGreaterEqual(float(ssim_fn.result), 0, f"SSIM value must be non-negative, got {ssim}.")
        self.assertLessEqual(float(ssim_fn.result), 1, f"SSIM value must be less or equal to 1, got {ssim}.")

    def test_version(self) -> None:
        self.assertGreaterEqual(API_VERSION, "1.2")

    def test_py_version(self) -> None:
        import platform

        py_version = Version(platform.python_version())
        self.assertGreaterEqual(py_version, "3.8")
