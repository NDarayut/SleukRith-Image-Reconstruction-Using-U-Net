"""
This scripts convert all the images stored in binary into jpg files
"""

import struct # Use to work with binary data
import numpy as np
import matplotlib.pyplot as plt

def load_image_data(data_file_path):
    """
    Loads image data from a binary file and returns a list of NumPy arrays.

    Parameters:
        data_file_path: The path to the binary file.

    Returns:
        A list of NumPy arrays representing the images.
    """

    with open(data_file_path, 'rb') as f:
        # Read width, height, and total number of samples
        width, height, total_num_samples = struct.unpack('>III', f.read(12))

        # Read image data
        image_data = np.fromfile(f, dtype=np.uint8).reshape(total_num_samples, height, width) # Reshape into 3D array

    return image_data

def save_images_as_jpg(image_data, output_folder, prefix="image"):
    """
    Saves a list of NumPy arrays as JPG images.

    Parameters:
        image_data: A list of NumPy arrays representing the images.
        output_folder: The path to the output folder.
        prefix: The prefix of the image name e.g "image_0, image_1...etc"
    """

    # Loop through the entire array in save it as jpg
    for i, image in enumerate(image_data):
        plt.imsave(f"{output_folder}/{prefix}_{i}.jpg", image, cmap='gray')
        print(f"image{i} saved")


data_file_path = 'train_data' # The binary file
output_folder = 'your/output/folder/path'

image_data = load_image_data(data_file_path)
save_images_as_jpg(image_data, output_folder)

print("done")