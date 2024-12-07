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
