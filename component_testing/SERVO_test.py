import time
import pigpio

# Define the four servo pins (change these to your preferred GPIOs)
SERVO_PINS = [12, 13, 18, 19]

# Pulse width boundaries for a typical servo (in microseconds)
MIN_PW = 500
MAX_PW = 2500
STEP = 1  # The increment step for pulse width when sweeping

pi = pigpio.pi()
if not pi.connected:
    print("Failed to connect to pigpio daemon. Start it with 'sudo pigpiod'.")
    exit(1)

# Initialize all servo pins
for pin in SERVO_PINS:
    pi.set_mode(pin, pigpio.OUTPUT)
    pi.set_servo_pulsewidth(pin, 0)  # Servo off initially

try:
    print("Sweeping servos. Press Ctrl+C to stop.")
    while True:
        # Sweep from min to max
        for pw in range(MIN_PW, MAX_PW + STEP, STEP):
            for pin in SERVO_PINS:
                pi.set_servo_pulsewidth(pin, pw)
            time.sleep(0.0005)
        # Sweep from max to min
        for pw in range(MAX_PW, MIN_PW - STEP, -STEP):
            for pin in SERVO_PINS:
                pi.set_servo_pulsewidth(pin, pw)
            time.sleep(0.0005)
except KeyboardInterrupt:
    pass
finally:
    # Turn off all servos
    for pin in SERVO_PINS:
        pi.set_servo_pulsewidth(pin, 0)
    pi.stop()
    print("Servos stopped.")
