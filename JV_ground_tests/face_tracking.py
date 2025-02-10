import re
import subprocess
import time
import pigpio  # For precise servo control

# --- Servo Configuration (using pulse widths directly) ---
SERVO_PIN = 13
SERVO_NEUTRAL = 1500   # Neutral position in µs (90° ≈ 1500µs)
SERVO_MIN = 1000       # Minimum pulse width (≈ 0°)
SERVO_MAX = 2000       # Maximum pulse width (≈ 180°)

# --- Proportional Controller Parameters ---
KP = 0.2         # Proportional gain (µs per unit error)
setpoint = 500   # Desired face Y coordinate
max_change = 100  # Limit maximum change per iteration (in µs)
deadband = 30    # Do nothing (no movement) when error is within ±30

# --- Initialize pigpio ---
pi = pigpio.pi()
if not pi.connected:
    exit()

def set_servo_pulsewidth(pulse_width):
    """Set the servo pulse width directly in microseconds."""
    pi.set_servo_pulsewidth(SERVO_PIN, int(pulse_width))

# --- Start the camera/detection process ---
cmd = [
    "rpicam-hello",
    "-t", "0",
    "-f",
    "--width", "640",
    "--height", "480",
    "--framerate", "15",
    "--post-process-file", "/usr/share/rpi-camera-assets/hailo_yolov5_personface.json",
    "--verbose"
]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

current_servo_pw = SERVO_NEUTRAL
set_servo_pulsewidth(current_servo_pw)

previous_time = time.time()

for line in process.stdout:
    line = line.strip()
    print("DEBUG:", line)

    # Look for a face detection line in the output
    match = re.search(r'Object: face\[2\] .*?@ (\d+),(\d+)', line)
    if match:
        _, y = map(int, match.groups())
        print(f"Detected Face at: Y={y}")

        # Compute the error; if the face is below the setpoint, error is positive or negative
        # depending on whether y < setpoint or y > setpoint
        error = y - setpoint
        print(f"Error: {error:.2f}")

        # Apply a deadband: if the absolute error is within 'deadband', do nothing
        if abs(error) < deadband:
            control_output = 0
            print(f"Within deadband ±{deadband}, not moving.")
        else:
            # Pure proportional control
            control_output = KP * error
            # Limit the maximum change in servo pulse width
            control_output = max(-max_change, min(max_change, control_output))

        # Update the servo pulse width if there's any control output
        if control_output != 0:
            current_servo_pw += control_output
            # Clamp the pulse width to the allowed range
            current_servo_pw = max(SERVO_MIN, min(SERVO_MAX, current_servo_pw))
            print(f"Moving Servo to: {current_servo_pw:.2f} µs")
            set_servo_pulsewidth(current_servo_pw)

        time.sleep(0.05)  # Small delay for movement stability

# Clean up
process.terminate()
pi.set_servo_pulsewidth(SERVO_PIN, 0)
pi.stop()
