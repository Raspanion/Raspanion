from smbus2 import SMBus
import time

# I2C Address for ICM-42688-P
ICM42688_ADDR = 0x69  # Adjust if necessary (0x69 if AD0 is pulled high)

# Registers
WHO_AM_I_ICM = 0x75
PWR_MGMT_0 = 0x4E
GYRO_CONFIG0 = 0x4F

# Accel data registers
ACCEL_DATA_X0 = 0x1F  # X0, X1, Y0, Y1, Z0, Z1

# Gyro data registers
GYRO_DATA_X0 = 0x25  # X0, X1, Y0, Y1, Z0, Z1

def initialize_icm42688(bus):
    # Check WHO_AM_I
    who_am_i = bus.read_byte_data(ICM42688_ADDR, WHO_AM_I_ICM)
    if who_am_i != 0x47:  # 0x47 is expected value
        print(f"Warning: ICM-42688 WHO_AM_I returned {hex(who_am_i)}, expected 0x47")

    # Set gyro and accel to low-noise mode:
    # 0x0F puts both in low noise mode
    bus.write_byte_data(ICM42688_ADDR, PWR_MGMT_0, 0x0F)

    # Set gyro configuration:
    # 0x00: default ±2000 dps full scale (adjust as needed)
    bus.write_byte_data(ICM42688_ADDR, GYRO_CONFIG0, 0x00)

    time.sleep(0.05)  # Allow stabilization

def read_accel(bus):
    # Read 6 bytes: X0,X1,Y0,Y1,Z0,Z1
    data = bus.read_i2c_block_data(ICM42688_ADDR, ACCEL_DATA_X0, 6)
    x = (data[1] << 8) | data[0]
    y = (data[3] << 8) | data[2]
    z = (data[5] << 8) | data[4]

    # Convert to signed 16-bit
    if x & 0x8000: x -= 1 << 16
    if y & 0x8000: y -= 1 << 16
    if z & 0x8000: z -= 1 << 16

    return x, y, z

def read_gyro(bus):
    # Read 6 bytes: X0,X1,Y0,Y1,Z0,Z1
    data = bus.read_i2c_block_data(ICM42688_ADDR, GYRO_DATA_X0, 6)
    x = (data[1] << 8) | data[0]
    y = (data[3] << 8) | data[2]
    z = (data[5] << 8) | data[4]

    # Convert to signed 16-bit
    if x & 0x8000: x -= 1 << 16
    if y & 0x8000: y -= 1 << 16
    if z & 0x8000: z -= 1 << 16

    return x, y, z

def main():
    with SMBus(1) as bus:  # Change if using a different I2C bus
        initialize_icm42688(bus)
        print("ICM-42688 initialized.")

        while True:
            ax, ay, az = read_accel(bus)
            gx, gy, gz = read_gyro(bus)

            print(f"Accel: X={ax}, Y={ay}, Z={az}")
            print(f"Gyro:  X={gx}, Y={gy}, Z={gz}")
            print("-" * 30)

            time.sleep(0.5)

if __name__ == "__main__":
    main()

