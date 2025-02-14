import smbus
import time

# Constants
I2C_ADDRESS = 0x69  # ICM-42688-P I2C address
GYRO_X_HIGH = 0x27  # Gyro Y-axis high byte (swapping with X-axis)
GYRO_X_LOW = 0x28   # Gyro Y-axis low byte
GYRO_Y_HIGH = 0x25  # Gyro X-axis high byte
GYRO_Y_LOW = 0x26   # Gyro X-axis low byte
GYRO_Z_HIGH = 0x29  # Gyro Z-axis high byte
GYRO_Z_LOW = 0x2A   # Gyro Z-axis low byte
PWR_MGMT0 = 0x4E
GYRO_CONFIG0 = 0x4F

# Initialize I2C bus
bus = smbus.SMBus(1)

def write_register(register, value):
    """Write a value to a specific register."""
    bus.write_byte_data(I2C_ADDRESS, register, value)

def read_register(register):
    """Read a value from a specific register."""
    return bus.read_byte_data(I2C_ADDRESS, register)

def read_gyro_axis(high_register, low_register):
    """Read and combine high and low bytes for a gyroscope axis."""
    high_byte = read_register(high_register)
    low_byte = read_register(low_register)
    value = (high_byte << 8) | low_byte
    if value & 0x8000:  # Convert to signed 16-bit
        value -= 65536
    return value

def configure_sensor():
    """Configure the gyroscope with appropriate settings."""
    write_register(PWR_MGMT0, 0x0F)  # Power on gyro and accel in low-noise mode
    write_register(GYRO_CONFIG0, 0x06)  # Set gyroscope ODR to 1kHz and full-scale range to ±2000 dps
    time.sleep(0.1)  # Wait for configuration to take effect

def main():
    """Main function to initialize the sensor and print gyro data continuously."""
    print("Configuring ICM-42688-P...")
    configure_sensor()

    print("Reading gyroscope data (press Ctrl+C to stop)...")
    try:
        while True:
            # Read gyro data
            gyro_x = -read_gyro_axis(GYRO_X_HIGH, GYRO_X_LOW)
            gyro_y = -read_gyro_axis(GYRO_Y_HIGH, GYRO_Y_LOW)
            gyro_z = -read_gyro_axis(GYRO_Z_HIGH, GYRO_Z_LOW)

            # Print the raw gyro values
            print(f"Gyro X: {gyro_x}, Gyro Y: {gyro_y}, Gyro Z: {gyro_z}")
            
            time.sleep(0.1)  # Delay to print approximately 10 times per second

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
