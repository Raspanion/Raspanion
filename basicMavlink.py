#!/usr/bin/env python3
"""
Basic MAVLink Companion Computer Script for Raspberry Pi
--------------------------------------------------------

This script demonstrates how to communicate with an ArduPilot-based flight controller
using MAVLink protocol via a companion computer (e.g., Raspberry Pi). It establishes
a connection over a serial interface, requests data streams, sends an arming command, 
and overrides RC channels sending a throttle PWM value of 1300 or 30%. The motors 
"should" disarm when stopping the script, i.e. pressing ctrl + C.

Setup Instructions:
- Install the pymavlink library:
    pip install pymavlink
- Configure the Raspberry Pi serial interface:
    sudo raspi-config
    # Navigate to Interface Options > Serial Port:
    # When prompted "Would you like a login shell to be accessible over serial?", select **No**.
    # When asked "Would you like the serial port hardware to be enabled?", select **Yes**.
- Ensure the UART interface is correctly connected to the flight controller.

**Safety Notice:** Modifying and sending commands to a drone can be dangerous.
Always ensure the drone is in a safe environment (e.g., props removed) when testing.
Have a plan to kill power if the motors do not stop when expected.

"""

import time
import math
from pymavlink import mavutil

# Constants
NUM_MSGS = 7  # Number of MAVLink messages to track
INTERVAL_US = 10000  # Message interval in microseconds (10 milliseconds)
IGNORE = 65535  # MAVLink ignore value for RC channels
SERIAL_PORT = '/dev/ttyAMA2'  # UART port (e.g., UART4 on Raspberry Pi)
BAUDRATE = 460800  # Serial communication baud rate

# MAVLink message IDs to request
MESSAGE_IDS = [
    mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT,
    mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
    mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
    mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
    mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
    mavutil.mavlink.MAVLINK_MSG_ID_HOME_POSITION,
]

# Flight mode mapping (custom_mode values to names)
FLIGHT_MODES = {
    0: "Stabilize",
    1: "Acro",
    2: "Altitude Hold",
    3: "Auto",
    4: "Guided",
    5: "Loiter",
    6: "Return to Launch",
    7: "Circle",
    9: "Land",
    11: "Drift",
    13: "Sport",
    16: "Position Hold",
}

# Initialize state tracking
last_msg_time = [0] * NUM_MSGS  # Timestamps for last received messages
stale_data_warning = False
stale_data_array = [False] * NUM_MSGS
time_to_print = 0
home_heading = 0


class LastState:
    """Class to store the last received MAVLink messages."""

    def __init__(self):
        self.heartbeat = None
        self.gps = None
        self.system = None
        self.attitude = None
        self.transmitter = None
        self.position = None
        self.home = None


# Instantiate the state tracking object
last = LastState()


def sanitize(value):
    """
    Ensure the value is an integer or IGNORE.

    :param value: The value to sanitize.
    :return: An integer value or IGNORE.
    """
    return IGNORE if value == IGNORE else int(value)


def request_data_stream():
    """
    Request MAVLink data streams for the specified message IDs.
    """
    for msg_id in MESSAGE_IDS:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,  # Confirmation
            msg_id,  # Message ID
            INTERVAL_US,  # Interval in microseconds
            0, 0, 0, 0, 0,  # Unused parameters
        )


def minimize_arming_checks(fewest):
    """
    Minimize or restore arming checks.

    :param fewest: True to minimize checks, False to restore default.
    """
    value = 8210 if fewest else 1
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        b'ARMING_CHECK',
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_UINT16,
    )


def set_flight_mode(flight_mode):
    """
    Set the flight mode of the drone.

    :param flight_mode: The custom_mode value corresponding to the desired flight mode.
    """
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,  # Confirmation
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        flight_mode,  # Desired flight mode
        0, 0, 0, 0, 0,  # Unused parameters
    )


def send_arm_command(arm, force):
    """
    Arm or disarm the drone.

    :param arm: True to arm, False to disarm.
    :param force: True to force arm/disarm even if safety checks fail.
    """
    param1 = 1.0 if arm else 0.0  # 1 to arm, 0 to disarm
    param2 = 21196 if force else 0  # Force arming/disarming if required

    print(f"{'Arming' if arm else 'Disarming'} the drone...")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,  # Confirmation
        param1,  # Arm/Disarm command
        param2,  # Force parameter
        0, 0, 0, 0, 0,  # Unused parameters
    )


def send_rc_override(roll, pitch, throttle, yaw):
    """
    Send RC channel override commands to control the drone.

    :param roll: Roll channel value or IGNORE.
    :param pitch: Pitch channel value or IGNORE.
    :param throttle: Throttle channel value or IGNORE.
    :param yaw: Yaw channel value or IGNORE.
    """
    roll = sanitize(roll)
    pitch = sanitize(pitch)
    throttle = sanitize(throttle)
    yaw = sanitize(yaw)

    # Uncomment the line below for debugging
    # print(f"Sending RC Override: roll={roll}, pitch={pitch}, throttle={throttle}, yaw={yaw}")

    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        roll, pitch, throttle, yaw,
        IGNORE, IGNORE, IGNORE, IGNORE,  # Channels 5-8 (ignored)
    )


def handle_mavlink_messages():
    """
    Handle incoming MAVLink messages from the flight controller.
    """
    global last_msg_time
    while True:
        msg = master.recv_match(blocking=False, timeout=0.1)
        if msg is None:
            break
        if msg.get_srcSystem() == 255:  # Skip messages from Ground Control Station
            continue
        msg_id = msg.get_msgId()
        if msg_id in MESSAGE_IDS:
            last_msg_time[MESSAGE_IDS.index(msg_id)] = time.time()

        # Update the last known state based on the message type
        if msg_id == mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT:
            last.heartbeat = msg
        elif msg_id == mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT:
            last.gps = msg
        elif msg_id == mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS:
            last.system = msg
        elif msg_id == mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE:
            last.attitude = msg
        elif msg_id == mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS:
            last.transmitter = msg
        elif msg_id == mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
            last.position = msg
        elif msg_id == mavutil.mavlink.MAVLINK_MSG_ID_HOME_POSITION:
            last.home = msg


def check_for_stale_data():
    """
    Check if any of the MAVLink messages have become stale.
    """
    global stale_data_warning, stale_data_array
    stale_data_warning = False
    current_time = time.time()
    for i in range(NUM_MSGS):
        if current_time - last_msg_time[i] > 2:
            stale_data_array[i] = True
            stale_data_warning = True
        else:
            stale_data_array[i] = False


def get_flight_mode_name(custom_mode):
    """
    Convert the custom_mode integer to a human-readable flight mode name.

    :param custom_mode: The custom_mode value from the heartbeat message.
    :return: The name of the flight mode.
    """
    return FLIGHT_MODES.get(custom_mode, f"Unknown ({custom_mode})")


def periodic_print(period):
    """
    Periodically print telemetry data for monitoring.

    :param period: The interval in seconds between prints.
    """
    global time_to_print
    if time.time() > time_to_print:
        time_to_print = time.time() + period
        if stale_data_warning:
            print("Warning: Stale Data Detected!")
        if last.heartbeat:
            armed = "Armed" if last.heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED else "Disarmed"
            print(f"Armed Status: {armed}")
            print(f"Flight Mode: {get_flight_mode_name(last.heartbeat.custom_mode)}")
        if last.system:
            print(f"Battery Voltage: {last.system.voltage_battery / 1000.0:.2f} V")
        if last.attitude:
            print(f"Roll Angle: {math.degrees(last.attitude.roll):.2f}°")
        if last.position:
            print(f"Altitude: {last.position.relative_alt / 1000.0:.2f} m")
        print(f"Heading to Home: {home_heading:.2f}°")
        print("-----------------------")


def calculate_heading_to_home():
    """
    Calculate the heading from the current position to the home position.
    """
    global home_heading
    if last.home and last.position:
        dx = last.home.longitude - last.position.lon
        dy = last.home.latitude - last.position.lat
        home_heading = math.degrees(math.atan2(dy, dx))
        if home_heading < 0:
            home_heading += 360


def main_loop():
    """
    The main control loop of the script.
    """
    while True:
        handle_mavlink_messages()
        check_for_stale_data()
        send_rc_override(IGNORE, IGNORE, 1300, 1500)  # Example RC override values
        periodic_print(2)  # Print telemetry every 2 seconds
        calculate_heading_to_home()
        time.sleep(0.01)  # Small delay to prevent CPU overutilization


if __name__ == "__main__":
    # Establish connection to the flight controller
    print(f"Connecting to flight controller on {SERIAL_PORT} at {BAUDRATE} baud...")
    master = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUDRATE)

    # Wait for the heartbeat message to find the system ID
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print(f"Heartbeat received from system (ID {master.target_system}), component (ID {master.target_component})")

    # Initial setup
    request_data_stream()
    minimize_arming_checks(True)
    set_flight_mode(0)  # Set to "Stabilize" mode
    send_arm_command(True, False)  # Arm the drone (use with caution)
    time.sleep(4)  # Allow time for initialization

    try:
        main_loop()
    except KeyboardInterrupt:
        # Disarm the drone safely before exiting
        print("Exiting and disarming the drone...")
        send_arm_command(False, True)
