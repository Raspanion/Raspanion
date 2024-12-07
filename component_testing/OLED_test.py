from smbus2 import SMBus, i2c_msg
from PIL import Image, ImageDraw, ImageFont
import time

# OLED I2C address (change this if your device has a different address)
DEVICE_ADDRESS = 0x3C

# OLED Display Size
WIDTH = 128
HEIGHT = 32
PI_BUS = 5

# Initialization commands for SSD1306 OLED
INIT_COMMANDS = [
    0xAE,  # Display off
    0x20,  # Set Memory Addressing Mode
    0x10,  # Page addressing mode
    0xB0,  # Set Page Start Address for Page Addressing Mode
    0xC8,  # COM Output Scan Direction
    0x00,  # Low Column Start Address
    0x10,  # High Column Start Address
    0x40,  # Start Line Address
    0x81,  # Set contrast control register
    0xFF,  # Max contrast
    0xA1,  # Segment re-map
    0xA6,  # Normal display
    0xA8,  # Set multiplex ratio
    0x1F,  # 1/32 duty (adjusted for 32-pixel height)
    0xA4,  # Output follows RAM content
    0xD3,  # Display offset
    0x00,  # No offset
    0xD5,  # Set display clock divide ratio/oscillator frequency
    0xF0,  # Max frequency
    0xD9,  # Set pre-charge period
    0x22,  
    0xDA,  # Set com pins hardware configuration
    0x02,  # Adjusted for 32-pixel height
    0xDB,  # Set vcomh
    0x20,  # 0.77xVcc
    0x8D,  # Enable charge pump regulator
    0x14,  
    0xAF   # Display ON
]

# Function to send commands to OLED
def initialize_oled(bus):
    for cmd in INIT_COMMANDS:
        bus.write_byte_data(DEVICE_ADDRESS, 0x00, cmd)  # Command mode
        time.sleep(0.01)  # Small delay for each command

# Create buffer and draw object
image = Image.new("1", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # You can replace this with a custom font if desired

# Function to display the buffer on the OLED
def display_image(bus, image):
    # Convert the image to binary format (1-bit color)
    pixel_data = list(image.getdata())
    # Loop through the 4 pages (32 pixels high / 8 pixels per page = 4 pages)
    for page in range(HEIGHT // 8):  # Adjust for 32-pixel height
        bus.write_byte_data(DEVICE_ADDRESS, 0x00, 0xB0 + page)  # Set page address
        bus.write_byte_data(DEVICE_ADDRESS, 0x00, 0x00)  # Set lower column address
        bus.write_byte_data(DEVICE_ADDRESS, 0x00, 0x10)  # Set higher column address
        for x in range(WIDTH):
            # Construct the byte for the 8-pixel vertical column
            byte = 0x00
            for bit in range(8):
                # Calculate the pixel position in the data array
                pixel = pixel_data[x + WIDTH * (page * 8 + bit)]
                byte |= (pixel & 0x01) << bit
            bus.write_byte_data(DEVICE_ADDRESS, 0x40, byte)

# Function to draw text and shapes
def draw_text_and_shapes():
    draw.rectangle((0, 0, WIDTH, HEIGHT), outline=0, fill=0)  # Clear screen
    
    draw.line((0, 0, WIDTH, HEIGHT), fill=255)
    draw.line((0, HEIGHT, WIDTH, 0), fill=255)
    draw.rectangle((20, 8, 40, 22), outline=255, fill=0)
    draw.ellipse((70, 8, 90, 22), outline=255, fill=0)
    draw.text((30, 22), "Hello, World!", font=font, fill=255)

# Main program
try:
    with SMBus(PI_BUS) as bus:
        print("Initializing OLED...")
        initialize_oled(bus)
        
        # Draw and display text and shapes
        draw_text_and_shapes()
        display_image(bus, image)
        
        print("Display updated. Check your OLED screen.")

except Exception as e:
    print(f"Error: {e}")



# from smbus2 import SMBus, i2c_msg
# import time

# # OLED I2C address (change this if your device has a different address)
# DEVICE_ADDRESS = 0x3C

# # Example initialization commands for SSD1306 OLED
# INIT_COMMANDS = [
#     0xAE,  # Display off
#     0x20,  # Set Memory Addressing Mode
#     0x10,  # Page addressing mode
#     0xB0,  # Set Page Start Address for Page Addressing Mode
#     0xC8,  # COM Output Scan Direction
#     0x00,  # Low Column Start Address
#     0x10,  # High Column Start Address
#     0x40,  # Start Line Address
#     0x81,  # Set contrast control register
#     0xFF,  # Max contrast
#     0xA1,  # Segment re-map
#     0xA6,  # Normal display
#     0xA8,  # Set multiplex ratio
#     0x3F,  # 1/64 duty
#     0xA4,  # Output follows RAM content
#     0xD3,  # Display offset
#     0x00,  # No offset
#     0xD5,  # Set display clock divide ratio/oscillator frequency
#     0xF0,  # Max frequency
#     0xD9,  # Set pre-charge period
#     0x22,  
#     0xDA,  # Set com pins hardware configuration
#     0x12,  
#     0xDB,  # Set vcomh
#     0x20,  # 0.77xVcc
#     0x8D,  # Enable charge pump regulator
#     0x14,  
#     0xAF   # Display ON
# ]

# # Send initialization commands to the OLED
# def initialize_oled(bus):
#     # Iterate over commands and send each one to the OLED
#     for cmd in INIT_COMMANDS:
#         bus.write_byte_data(DEVICE_ADDRESS, 0x00, cmd)  # 0x00 is the command mode for SSD1306
#         time.sleep(0.01)  # Small delay to allow each command to process

# # Draw an interesting pattern to check if the display works
# def draw_test_pattern(bus):
#     for page in range(0xB0, 0xB8):  # Loop through pages for SSD1306
#         bus.write_byte_data(DEVICE_ADDRESS, 0x00, page)  # Set the page address
#         # Create a checkerboard pattern by alternating bytes 0xAA and 0x55
#         for col in range(128):  # Width of OLED is usually 128 pixels
#             if (col // 8) % 2 == 0:
#                 bus.write_byte_data(DEVICE_ADDRESS, 0x40, 0xAA)  # Pattern part 1
#             else:
#                 bus.write_byte_data(DEVICE_ADDRESS, 0x40, 0x55)  # Pattern part 2

# # Main program
# try:
#     # Open I2C bus 4
#     with SMBus(4) as bus:
#         print("Initializing OLED...")
#         initialize_oled(bus)
#         print("Drawing test pattern...")
#         draw_test_pattern(bus)
#         print("Pattern displayed. Check your OLED screen.")

# except Exception as e:
#     print(f"Error: {e}")




# import time
# import board
# import busio
# from adafruit_ssd1306 import SSD1306_I2C
# from PIL import Image, ImageDraw, ImageFont

# # Set up I2C and OLED
# i2c = busio.I2C(board.SCL, board.SDA, bus=4)
# oled = SSD1306_I2C(128, 32, i2c)  # Adjust width and height to your OLED's specs

# # Clear display
# oled.fill(0)
# oled.show()

# # Create an image buffer
# width = oled.width
# height = oled.height
# image = Image.new("1", (width, height))

# # Create a drawing object
# draw = ImageDraw.Draw(image)

# # Load a default font
# font = ImageFont.load_default()

# # Draw test patterns
# draw.rectangle((0, 0, width-1, height-1), outline=255, fill=0)  # Border
# draw.text((10, 10), "OLED Test", font=font, fill=255)
# draw.line((0, height - 1, width - 1, 0), fill=255)  # Diagonal line

# # Display image
# oled.image(image)
# oled.show()

# time.sleep(15)  # Keeps display on for 5 seconds before clearing
# oled.fill(0)
# oled.show()
