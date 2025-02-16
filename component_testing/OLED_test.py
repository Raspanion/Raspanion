from smbus2 import SMBus
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
    0x1F,  # 1/32 duty (for 32-pixel height)
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

def initialize_oled(bus):
    for cmd in INIT_COMMANDS:
        bus.write_byte_data(DEVICE_ADDRESS, 0x00, cmd)  # Command mode
        time.sleep(0.01)

# Create image buffer and drawing object
image = Image.new("1", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(image)

# Load a slightly smaller TrueType font (size 20)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except IOError:
    print("Custom font not found, falling back to default font.")
    font = ImageFont.load_default()

def display_text(bus, text):
    # Clear the screen
    draw.rectangle((0, 0, WIDTH, HEIGHT), outline=0, fill=0)
    
    # Compute the bounding box for the text.
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text using the bounding box offsets.
    x = (WIDTH - text_width) // 2 - bbox[0]
    y = (HEIGHT - text_height) // 2 - bbox[1]
    
    draw.text((x, y), text, font=font, fill=255)
    
    # Convert image to binary data and send it to the OLED.
    pixel_data = list(image.getdata())
    for page in range(HEIGHT // 8):
        bus.write_byte_data(DEVICE_ADDRESS, 0x00, 0xB0 + page)  # Set page address
        bus.write_byte_data(DEVICE_ADDRESS, 0x00, 0x00)           # Set lower column address
        bus.write_byte_data(DEVICE_ADDRESS, 0x00, 0x10)           # Set higher column address
        for col in range(WIDTH):
            byte = 0x00
            for bit in range(8):
                pixel = pixel_data[col + WIDTH * (page * 8 + bit)]
                byte |= (pixel & 0x01) << bit
            bus.write_byte_data(DEVICE_ADDRESS, 0x40, byte)

try:
    with SMBus(PI_BUS) as bus:
        print("Initializing OLED...")
        initialize_oled(bus)
        
        # Display "Raspanion" with the slightly smaller font.
        display_text(bus, "Raspanion")
        print("Display updated. Check your OLED screen.")

except Exception as e:
    print(f"Error: {e}")

