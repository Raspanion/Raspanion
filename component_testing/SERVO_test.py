# Import necessary libraries
import pigpio  # Library to control GPIO pins using the pigpio daemon
import time    # Library to add delays in the code

# Define the GPIO pin connected to the servo
SERVO_PIN = 13  # Change this to the GPIO pin number you are using for the servo

# Define the middle pulse width for the servo (in microseconds)
# Typical range for servo pulse width is 500 to 2500 microseconds
MIDDLE_PW = 1500  # 1500 microseconds corresponds to the middle position

# Initialize the pigpio library and connect to the pigpio daemon
pi = pigpio.pi()
if not pi.connected:
    # If pigpio is not running, display an error and exit the program
    print("Failed to connect to pigpio daemon. Start it with 'sudo pigpiod'.\n"
          "Note: To avoid typing 'sudo pigpiod' every time, run:\n"
          "sudo systemctl enable pigpiod\n"
          "sudo systemctl start pigpiod")
    exit(1)

try:
    # Inform the user that the servo is moving to the middle position
    print("Moving GPIO 12 servo to middle position. Press Ctrl+C to stop.")

    # Set the servo to the middle position by specifying the pulse width
    pi.set_servo_pulsewidth(SERVO_PIN, MIDDLE_PW)

    # Keep the program running so the servo holds its position
    while True:
        time.sleep(1)  # Add a small delay to prevent excessive CPU usage

except KeyboardInterrupt:
    # Handle the Ctrl+C keyboard interrupt gracefully
    pass

finally:
    # Turn off the servo by setting the pulse width to 0 (stops sending signals)
    pi.set_servo_pulsewidth(SERVO_PIN, 0)

    # Stop the pigpio connection
    pi.stop()

    # Inform the user that the program has stopped
    print("Servo stopped.")
