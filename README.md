# DronePiShoe
Pi code to control HAT and interact with flight controllers












Replace the text in /boot/config.txt on your Pi with the following text:

# For more options and information see:
# http://rpf.io/configtxt

# Enable composite video output (TV out)
enable_tvout=1
sdtv_mode=2       # Set to PAL mode (Europe and others)
sdtv_aspect=3     # Set 16:9 aspect ratio for composite video

# Set GPU memory allocation (128 MB for better graphics support)
gpu_mem=128

# Enable I2C interfaces for additional peripherals
dtparam=i2c_arm=on
dtoverlay=i2c4,pins_6_7   # I2C on GPIO pins 6 and 7
dtoverlay=i2c5,pins_10_11 # I2C on GPIO pins 10 and 11

# Enable audio
dtparam=audio=on

# Automatically load overlays for detected cameras
camera_auto_detect=1
display_auto_detect=1

# Disable compensation for displays with overscan
disable_overscan=1

# Set up overlays for ArduCam and IMX519 camera modules
dtoverlay=arducam-pivariety
dtoverlay=imx519
dtoverlay=imx519,cam0

# Set up serial port(s)
enable_uart=1
dtoverlay=uart0
dtoverlay=uart3
dtoverlay=uart4

# Enable DRM (Direct Rendering Manager) for graphics acceleration
dtoverlay=vc4-fkms-v3d  # Use FKMS for compatibility with composite output

# Specific settings for Compute Module 4
[cm4]
otg_mode=1  # Enable USB OTG mode on the CM4's built-in XHCI USB controller

# Specific settings for Pi 4
[pi4]
arm_boost=1  # Enable max CPU speed for Pi 4

