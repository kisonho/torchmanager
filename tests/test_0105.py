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
