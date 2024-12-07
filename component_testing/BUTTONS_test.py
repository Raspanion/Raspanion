import RPi.GPIO as GPIO
import time

# Use BCM numbering
GPIO.setmode(GPIO.BCM)

# Dictionary of buttons and their corresponding GPIO pins
buttons = {
    "right": 23, # (this also boot button)
    "left": 24,
    "down": 25,
    "enter": 26,
    "up": 27
}

# Set up each button as an input with internal pull-up resistor
for name, pin in buttons.items():
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Press Ctrl+C to exit.")
try:
    while True:
        for name, pin in buttons.items():
            # Button pressed if input is LOW
            if GPIO.input(pin) == GPIO.LOW:
                print(f"Button {name} pressed.")
        time.sleep(0.1)  # Slight delay to debounce and reduce CPU usage
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
