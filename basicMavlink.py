import time
import math
from pymavlink import mavutil

# Constants
NUM_MSGS = 7
message_ids = [
    mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT,
    mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
    mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
    mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
    mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
    mavutil.mavlink.MAVLINK_MSG_ID_HOME_POSITION
]

staleDataWarning = False
staleDataArray = [False] * NUM_MSGS
last_msg_time = [0] * NUM_MSGS
timeToPrint = 0
home_heading = 0

SYSTEM_ID = 255
COMPONENT_ID = 2
TARGET_SYSTEM = 1
TARGET_COMPONENT = 1
CONFIRMATION = 0
IGNORE = 65535
INTERVAL_US = 10000  # 10 milliseconds

# Connect to the FC
serial_port = '/dev/ttyAMA1'  # UART4
baudrate = 460800

master = mavutil.mavlink_connection(serial_port, baud=baudrate)

class LastState:
    def __init__(self):
        self.heartbeat = None
        self.gps = None
        self.system = None
        self.attitude = None
        self.transmitter = None
        self.position = None
        self.home = None

last = LastState()

def requestDataStream():
    for msg_id in message_ids:
        master.mav.command_long_send(
            master.target_system,  # target_system
            master.target_component,  # target_component
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,  # command
            0,  # confirmation
            msg_id,  # param1: message id
            INTERVAL_US,  # param2: interval in microseconds
            0, 0, 0, 0, 0  # param3 ~ param7 not used
        )

def minimizeArmingChecks(fewest):
    value = 8210 if fewest else 1
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        b'ARMING_CHECK',
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_UINT16
    )

def setFlightMode(flightMode):
    master.mav.command_long_send(
        master.target_system,  # target_system
        master.target_component,  # target_component
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,  # command
        0,  # confirmation
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,  # param1
        flightMode,  # param2: custom mode
        0, 0, 0, 0, 0  # param3 ~ param7 not used
    )

def sendArmCommand(arm, force):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,  # confirmation
        1 if arm else 0,  # param1: 1 to arm, 0 to disarm
        21196 if force else 0,  # param2: force
        0, 0, 0, 0, 0
    )

def handleMAVLinkMessages():
    global last_msg_time
    while True:
        msg = master.recv_match(blocking=False, timeout=0.1)
        if msg is None:
            break
        # Skip messages from GCS (System ID = 255)
        if msg.get_srcSystem() == 255:
            continue
        msg_id = msg.get_msgId()
        if msg_id in message_ids:
            msgIndex = message_ids.index(msg_id)
            last_msg_time[msgIndex] = time.time()
        # Now handle the messages
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

def checkForStaleData():
    global staleDataWarning, staleDataArray
    staleDataWarning = False
    current_time = time.time()
    for i in range(NUM_MSGS):
        if current_time - last_msg_time[i] > 2:
            staleDataArray[i] = True
            staleDataWarning = True
        else:
            staleDataArray[i] = False

def sendRCOverride(roll, pitch, throttle, yaw):
    def ignore_value(val):
        return None if val == IGNORE else val

    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        ignore_value(roll),
        ignore_value(pitch),
        ignore_value(throttle),
        ignore_value(yaw),
        None, None, None, None  # channels 5-8
    )

def getFlightMode(custom_mode):
    flight_modes = {
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
        16: "Pos Hold"
    }
    return flight_modes.get(custom_mode, "Unknown")

def periodicPrint(period):
    global timeToPrint
    current_time = time.time()
    if current_time > timeToPrint:
        timeToPrint = current_time + period
        if staleDataWarning:
            print("Stale Data")
        else:
            print("Armed:", "Armed" if last.heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED else "Disarmed")
            print("Flight Mode:", getFlightMode(last.heartbeat.custom_mode))
            print("Voltage:", last.system.voltage_battery / 1000.0)
            print("Roll:", last.attitude.roll * 180.0 / math.pi)
            print("Altitude:", last.position.relative_alt / 1000.0)
            print("Heading to Home:", home_heading)
        print("-----------------------")

def calculateHeadingToHome():
    global home_heading
    if last.home is not None:
        home_heading = math.atan2(last.home.y, last.home.x) * (180.0 / math.pi)
        if home_heading < 0:
            home_heading += 360.0

def sendYawEstimate(yaw_radians):
    if last.attitude is not None:
        usec = int(last.attitude.time_boot_ms * 1000)
    else:
        usec = int(time.time() * 1e6)
    master.mav.vision_position_estimate_send(
        usec,  # Timestamp in microseconds
        10.0,  # X position
        20.0,  # Y position
        0.0,   # Z position
        0.0,   # Roll
        0.0,   # Pitch
        yaw_radians  # Yaw
    )

def sendVisionSpeedEstimate(vx, vy, vz):
    if last.attitude is not None:
        usec = int(last.attitude.time_boot_ms * 1000)
    else:
        usec = int(time.time() * 1e6)
    master.mav.vision_speed_estimate_send(
        usec,  # Timestamp in microseconds
        vx,
        vy,
        vz
    )

def main_loop():
    while True:
        handleMAVLinkMessages()
        checkForStaleData()
        sendRCOverride(IGNORE, IGNORE, 1300, 1500)
        periodicPrint(2)  # 2 seconds
        calculateHeadingToHome()
        sendYawEstimate(0)
        sendVisionSpeedEstimate(-1, -1, 0)
        time.sleep(0.01)

if __name__ == "__main__":
    # Wait for a heartbeat to find the target system ID and component ID
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Heartbeat received from system (system %u component %u)" % (master.target_system, master.target_component))

    # Now send the initial setup commands
    requestDataStream()
    minimizeArmingChecks(True)  # minimize arming checks
    setFlightMode(0)  # set to stabilize mode
    # sendArmCommand(True, False)  # arm the drone if needed
    time.sleep(4)  # Allow time for the drone to initialize

    try:
        main_loop()
    except KeyboardInterrupt:
        print("Exiting...")
