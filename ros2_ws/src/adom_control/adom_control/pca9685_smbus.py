import time


class LinuxPca9685:
    """Minimal PCA9685 driver backed by Ubuntu's python3-smbus package."""

    MODE1 = 0x00
    MODE2 = 0x01
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    OSCILLATOR_HZ = 25_000_000.0

    def __init__(self, bus_number, address, frequency_hz):
        try:
            import smbus
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "python3-smbus is not installed; run: sudo apt install python3-smbus"
            ) from exc

        self.address = int(address)
        self.frequency_hz = float(frequency_hz)
        if not 24.0 <= self.frequency_hz <= 1526.0:
            raise ValueError("PCA9685 frequency must be between 24 and 1526 Hz")

        self.bus = smbus.SMBus(int(bus_number))
        # Reading MODE1 verifies the configured bus/address before any writes.
        old_mode = self.bus.read_byte_data(self.address, self.MODE1)
        prescale = round(
            self.OSCILLATOR_HZ / (4096.0 * self.frequency_hz) - 1.0
        )
        sleep_mode = (old_mode & 0x7F) | 0x10
        self.bus.write_byte_data(self.address, self.MODE1, sleep_mode)
        self.bus.write_byte_data(self.address, self.PRESCALE, int(prescale))
        self.bus.write_byte_data(self.address, self.MODE1, old_mode)
        time.sleep(0.005)
        # RESTART | auto-increment. OUTDRV enables the normal totem-pole output.
        self.bus.write_byte_data(self.address, self.MODE1, old_mode | 0xA0)
        mode2 = self.bus.read_byte_data(self.address, self.MODE2)
        self.bus.write_byte_data(self.address, self.MODE2, mode2 | 0x04)

    def set_pulse_us(self, channel, pulse_us):
        channel = int(channel)
        if not 0 <= channel <= 15:
            raise ValueError("PCA9685 channel must be between 0 and 15")
        count = round(float(pulse_us) * self.frequency_hz * 4096.0 / 1_000_000.0)
        off_count = int(max(1, min(4095, count)))
        register = self.LED0_ON_L + 4 * channel
        self.bus.write_i2c_block_data(
            self.address,
            register,
            [0x00, 0x00, off_count & 0xFF, (off_count >> 8) & 0x0F],
        )

    def close(self):
        self.bus.close()
