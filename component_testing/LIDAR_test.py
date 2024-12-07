import serial
import time

# Adjust the serial port if necessary; this is typically correct for UART0 and UART3 once enabled.
SERIAL_PORT = "/dev/ttyAMA0" # UART0
# SERIAL_PORT = "/dev/ttyAMA1" # UART3
BAUDRATE = 115200

def read_tfmini_frame(ser):
    """
    Reads one valid TF Mini data frame.
    Blocks until a full valid frame is received.
    Returns (distance, strength) as integers.
    """
    # Wait for start bytes 0x59 0x59
    while True:
        byte1 = ser.read(1)
        if byte1 == b'\x59':  # Possible start
            byte2 = ser.read(1)
            if byte2 == b'\x59':
                # We have found the start of a frame
                break
    # Now read the remaining 7 bytes
    frame = ser.read(7)

    # frame: Distance_L, Distance_H, Strength_L, Strength_H, Reserved, Quality/Temp, Checksum
    distance = frame[0] + (frame[1] << 8)
    strength = frame[2] + (frame[3] << 8)

    # Checksum (verify if needed)
    # Checksum is the low byte of the sum of bytes 0-7 (including the two start bytes)
    # sum_data = 0x59 + 0x59 + sum of frame[0..6]
    # This script will assume data is correct. To fully verify:
    sum_data = 0x59 + 0x59 + sum(frame[:6])
    checksum = sum_data & 0xFF
    if checksum != frame[6]:
        # If checksum fails, we can discard this frame and try again.
        return None, None
    return distance, strength

def main():
    # Open the serial port
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(0.1)  # Allow some time to initialize

    try:
        print("Reading TF Mini LiDAR data from UART3...")
        while True:
            distance, strength = read_tfmini_frame(ser)
            if distance is not None and strength is not None:
                print(f"Distance: {distance} cm, Strength: {strength}")
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
