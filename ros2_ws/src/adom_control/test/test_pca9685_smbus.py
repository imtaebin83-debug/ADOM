import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from adom_control.pca9685_smbus import LinuxPca9685


class FakeBus:
    def __init__(self):
        self.byte_writes = []
        self.block_writes = []

    def read_byte_data(self, _address, register):
        return 0x01 if register == LinuxPca9685.MODE1 else 0x04

    def write_byte_data(self, address, register, value):
        self.byte_writes.append((address, register, value))

    def write_i2c_block_data(self, address, register, data):
        self.block_writes.append((address, register, data))

    def close(self):
        pass


def make_driver():
    bus = FakeBus()
    module = SimpleNamespace(SMBus=lambda _number: bus)
    with patch.dict(sys.modules, {"smbus": module}):
        driver = LinuxPca9685(7, 0x40, 50.0)
    return driver, bus


class LinuxPca9685Test(unittest.TestCase):
    def test_configures_50_hz_and_writes_1500_us_to_channel_1(self):
        driver, bus = make_driver()
        driver.set_pulse_us(1, 1500.0)

        self.assertIn((0x40, LinuxPca9685.PRESCALE, 121), bus.byte_writes)
        self.assertEqual(
            bus.block_writes[-1], (0x40, 0x0A, [0x00, 0x00, 0x33, 0x01])
        )

    def test_rejects_invalid_channel(self):
        driver, _bus = make_driver()

        with self.assertRaisesRegex(ValueError, "between 0 and 15"):
            driver.set_pulse_us(16, 1500.0)
