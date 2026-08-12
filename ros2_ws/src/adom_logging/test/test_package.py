import unittest


class AdomLoggingPackageTest(unittest.TestCase):
    def test_package_imports(self):
        import adom_logging

        self.assertTrue(adom_logging.__doc__)
