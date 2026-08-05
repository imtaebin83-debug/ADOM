import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from adom_control.pca9685_smbus import LinuxPca9685


class FakeBus:
    def __init__(self, mode1=0x01, mode2=0x04):
        self.byte_writes = []
        self.block_writes = []
        self.registers = {
            LinuxPca9685.MODE1: mode1,
            LinuxPca9685.MODE2: mode2,
            LinuxPca9685.PRESCALE: 0x1E,
        }
        self.closed = False

    def read_byte_data(self, _address, register):
        return self.registers.get(register, 0x00)

    def write_byte_data(self, address, register, value):
        self.byte_writes.append((address, register, value))
        self.registers[register] = value

    def write_i2c_block_data(self, address, register, data):
        self.block_writes.append((address, register, data))

    def close(self):
        self.closed = True


def make_driver(mode1=0x01, mode2=0x04):
    bus = FakeBus(mode1, mode2)
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

    def test_wakes_oscillator_when_previous_process_left_sleep_set(self):
        _driver, bus = make_driver(mode1=0x31)

        mode1_writes = [
            value
            for _address, register, value in bus.byte_writes
            if register == LinuxPca9685.MODE1
        ]
        self.assertEqual(mode1_writes, [0x11, 0x01, 0xA1])
        self.assertEqual(mode1_writes[-1] & 0x10, 0)

    def test_replaces_stale_mode2_and_global_output_state(self):
        _driver, bus = make_driver(mode2=0x1F)

        self.assertIn(
            (0x40, LinuxPca9685.MODE2, LinuxPca9685.MODE2_SERVO_OUTPUT),
            bus.byte_writes,
        )
        self.assertIn(
            (0x40, LinuxPca9685.ALL_LED_ON_L, [0x00, 0x00, 0x00, 0x00]),
            bus.block_writes,
        )

    def test_rejects_sticky_external_clock_mode(self):
        bus = FakeBus(mode1=0x51)
        module = SimpleNamespace(SMBus=lambda _number: bus)

        with patch.dict(sys.modules, {"smbus": module}):
            with self.assertRaisesRegex(RuntimeError, "EXTCLK"):
                LinuxPca9685(7, 0x40, 50.0)

        self.assertTrue(bus.closed)
