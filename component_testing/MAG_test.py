from smbus2 import SMBus
import time

# I2C address of LIS2MDLTR
LIS2MDLTR_ADDR = 0x1E

# Register addresses
WHO_AM_I_LIS = 0x4F
STATUS_REG = 0x67
CTRL_REG1_LIS = 0x60
CTRL_REG2_LIS = 0x61
CTRL_REG3_LIS = 0x62
CTRL_REG4_LIS = 0x63
MAG_OUTX_L = 0x68

def initialize_lis2mdltr(bus):
    # Soft reset the sensor
    bus.write_byte_data(LIS2MDLTR_ADDR, CTRL_REG2_LIS, 0x40)
    time.sleep(0.1)

    # Verify WHO_AM_I
    who_am_i = bus.read_byte_data(LIS2MDLTR_ADDR, WHO_AM_I_LIS)
    if who_am_i != 0x40:
        print(f"Warning: WHO_AM_I returned {hex(who_am_i)}, expected 0x40")

    # Basic config: ODR=10Hz, normal mode
    # CTRL_REG1_LIS: 0x10 = ODR=10Hz, Low-power mode off
    bus.write_byte_data(LIS2MDLTR_ADDR, CTRL_REG1_LIS, 0x10)

    # CTRL_REG2_LIS: Leave default (0x00)
    bus.write_byte_data(LIS2MDLTR_ADDR, CTRL_REG2_LIS, 0x00)

    # CTRL_REG3_LIS: Continuous-conversion mode (0x00)
    bus.write_byte_data(LIS2MDLTR_ADDR, CTRL_REG3_LIS, 0x00)

    # CTRL_REG4_LIS: Default (0x00)
    bus.write_byte_data(LIS2MDLTR_ADDR, CTRL_REG4_LIS, 0x00)

    time.sleep(0.05)  # Allow time for stabilization

def read_magnetometer(bus):
    # Check status to ensure new data is available
    status = bus.read_byte_data(LIS2MDLTR_ADDR, STATUS_REG)
    if not (status & 0x08):
        # No new data ready
        return None, None, None

    data = bus.read_i2c_block_data(LIS2MDLTR_ADDR, MAG_OUTX_L, 6)
    x = (data[1] << 8) | data[0]
    y = (data[3] << 8) | data[2]
    z = (data[5] << 8) | data[4]

    # Convert to signed 16-bit
    if x & 0x8000: x -= 1 << 16
    if y & 0x8000: y -= 1 << 16
    if z & 0x8000: z -= 1 << 16

    return x, y, z

def main():
    # Adjust the bus number if your sensor is on a different bus.
    with SMBus(5) as bus:
        initialize_lis2mdltr(bus)
        print("LIS2MDLTR initialized.")

        while True:
            mx, my, mz = read_magnetometer(bus)
            if mx is not None:
                print(f"Mag: X={mx}, Y={my}, Z={mz}")
                print("-" * 30)
            else:
                print("No new data available")
            time.sleep(0.5)

if __name__ == "__main__":
    main()
