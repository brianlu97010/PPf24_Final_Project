import numpy as np
from PIL import Image
import os

def create_test_image(width, height, filename):
    # Create a gradient pattern
    x = np.linspace(0, 255, width)
    y = np.linspace(0, 255, height)
    X, Y = np.meshgrid(x, y)
    
    # Create RGB channels
    r = X
    g = Y
    b = (X + Y) / 2
    
    # Combine channels
    img_array = np.stack((r, g, b), axis=-1).astype(np.uint8)
    
    # Create and save image
    img = Image.fromarray(img_array)
    img.save(filename, 'BMP')
    
    # Get file size in MB
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"Created {filename}: {width}x{height} pixels, {file_size:.2f} MB")

# Create output directory if it doesn't exist
os.makedirs("test_images", exist_ok=True)

# Define test sizes (width, height)
sizes = [
    (640, 480),      # VGA
    (1280, 720),     # 720p
    (1920, 1080),    # 1080p
    (2560, 1440),    # 2K
    (3840, 2160),    # 4K
    (5120, 2880),    # 5K
]

# Generate images
for i, (width, height) in enumerate(sizes):
    filename = f"test_images/test_{width}x{height}.bmp"
    create_test_image(width, height, filename)
