"""
This scripts allow you to view the image straight from the binary file
"""
import struct # Use to unpack binary data
import numpy as np
import matplotlib.pyplot as plt

data_file = r'train_data'

with open(data_file, 'rb') as f:
    # Read width, height, and number of samples
    width, height, nb_samples = struct.unpack('>III', f.read(12)) # Read the first 12 bytes

    # Read image data
    data = np.fromfile(f, dtype=np.uint8).reshape(nb_samples, height, width) # Reshape it into a 3D array

# Choose an image index to visualize
image_index = 0

# Display the image
plt.imshow(data[image_index], cmap='gray')
plt.title(f'Image {image_index}')
plt.show()