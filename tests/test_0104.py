from unittest import TestCase


class Test0104(TestCase):
    def test_cosine_similarity(self) -> None:
        from torchmanager.metrics import CosineSimilarity
        from torchmanager_core import torch

        # initialize random data
        x = torch.rand((1, 3, 256, 256))
        y = torch.rand_like(x)
        cosine_similarity_fn = CosineSimilarity()

        # calculate cosine similarity
        cosine_similarity = cosine_similarity_fn(x, y)
        self.assertGreaterEqual(float(cosine_similarity_fn.result), -1.0, f"Cosine similarity value must be greater than or equal to -1.0, got {cosine_similarity}.")
        self.assertLessEqual(float(cosine_similarity_fn.result), 1.0, f"Cosine similarity value must be less than or equal to 1.0, got {cosine_similarity}.")

    def test_py_version(self) -> None:
        import platform
        major, minor, _ = platform.python_version_tuple()
        self.assertGreaterEqual(int(major), 3)
        self.assertGreaterEqual(int(minor), 10)

    def test_version(self) -> None:
        from torchmanager_core import API_VERSION
        self.assertGreaterEqual(API_VERSION, "1.4")
