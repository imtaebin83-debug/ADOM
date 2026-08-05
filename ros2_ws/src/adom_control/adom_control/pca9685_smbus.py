import time


class LinuxPca9685:
    """Minimal PCA9685 driver backed by Ubuntu's python3-smbus package."""

    MODE1 = 0x00
    MODE2 = 0x01
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    ALL_LED_ON_L = 0xFA
    OSCILLATOR_HZ = 25_000_000.0
    MODE1_RESTART = 0x80
    MODE1_EXTCLK = 0x40
    MODE1_AI = 0x20
    MODE1_SLEEP = 0x10
    MODE1_ALLCALL = 0x01
    MODE2_SERVO_OUTPUT = 0x04

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
        if old_mode & self.MODE1_EXTCLK:
            self.bus.close()
            raise RuntimeError(
                "PCA9685 EXTCLK is set but this driver requires the internal clock; "
                "power-cycle the PCA9685 to clear the sticky EXTCLK bit"
            )

        prescale = round(
            self.OSCILLATOR_HZ / (4096.0 * self.frequency_hz) - 1.0
        )
        # Configure from a deterministic base instead of preserving mode bits
        # left by another process.  PRE_SCALE is writable only while asleep.
        sleep_mode = self.MODE1_SLEEP | self.MODE1_ALLCALL
        awake_mode = self.MODE1_ALLCALL
        self.bus.write_byte_data(self.address, self.MODE1, sleep_mode)
        self.bus.write_byte_data(self.address, self.PRESCALE, int(prescale))
        self.bus.write_byte_data(self.address, self.MODE1, awake_mode)
        # The data sheet requires at least 500 us between clearing SLEEP and
        # requesting RESTART.  Five milliseconds provides ample margin.
        time.sleep(0.005)
        self.bus.write_byte_data(
            self.address,
            self.MODE1,
            awake_mode | self.MODE1_RESTART | self.MODE1_AI,
        )
        # Non-inverted, update-on-STOP, totem-pole output.  Do not preserve
        # inversion or OE behavior configured by a previous process.
        self.bus.write_byte_data(
            self.address, self.MODE2, self.MODE2_SERVO_OUTPUT
        )
        # A global FULL_OFF bit overrides every per-channel PWM register.
        self.bus.write_i2c_block_data(
            self.address, self.ALL_LED_ON_L, [0x00, 0x00, 0x00, 0x00]
        )

        verified_mode1 = self.bus.read_byte_data(self.address, self.MODE1)
        verified_mode2 = self.bus.read_byte_data(self.address, self.MODE2)
        verified_prescale = self.bus.read_byte_data(self.address, self.PRESCALE)
        if verified_mode1 & self.MODE1_SLEEP:
            self.bus.close()
            raise RuntimeError(
                "PCA9685 oscillator remained asleep after initialization"
            )
        if not verified_mode1 & self.MODE1_AI:
            self.bus.close()
            raise RuntimeError("PCA9685 auto-increment did not enable")
        if verified_mode2 != self.MODE2_SERVO_OUTPUT:
            self.bus.close()
            raise RuntimeError(
                "PCA9685 MODE2 verification failed: expected 0x%02x, got 0x%02x"
                % (self.MODE2_SERVO_OUTPUT, verified_mode2)
            )
        if verified_prescale != int(prescale):
            self.bus.close()
            raise RuntimeError(
                "PCA9685 PRE_SCALE verification failed: expected %d, got %d"
                % (int(prescale), verified_prescale)
            )

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
