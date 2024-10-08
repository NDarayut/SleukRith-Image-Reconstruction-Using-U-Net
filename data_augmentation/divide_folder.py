"""
This scripts divide the images into three folder for data augmentation.
Each folder is a level of augmentation e.g Folder1->Level 1, Folder2->Level 2
"""

import os
import random
from PIL import Image

random.seed(0)

# Path to the source directory where the original images are located
source_dir = r"testset_original"

# Paths to the three destination folders
dest_dir1 = r"set_1"
dest_dir2 = r"set_2"
dest_dir3 = r"set_3"

# Ensure destination directories exist, or create them
os.makedirs(dest_dir1, exist_ok=True)
os.makedirs(dest_dir2, exist_ok=True)
os.makedirs(dest_dir3, exist_ok=True)

# Get a list of all the image files in the source directory
all_images = os.listdir(source_dir)

# Shuffle the list of images and randomly select 30,000 images
random.shuffle(all_images)
random.shuffle(all_images)
random.shuffle(all_images)
selected_images = all_images[:30000]

# Divide selected images into 3 groups, each containing 10,000 images
images_for_folder1 = selected_images[:10000]
images_for_folder2 = selected_images[10000:20000]
images_for_folder3 = selected_images[20000:30000]

# Function to convert an image to grayscale and save it to the destination folder
def convert_and_move_images(image_list, dest_folder):
    for image in image_list:
        source_path = os.path.join(source_dir, image)
        dest_path = os.path.join(dest_folder, image)

        # Open the image and convert it to grayscale
        try:
            img = Image.open(source_path).convert('L')  # Convert to grayscale
            img.save(dest_path)  # Save the grayscale image to the destination
        except Exception as e:
            print(f"Error processing {image}: {e}")

# Convert and move the images to each respective folder
convert_and_move_images(images_for_folder1, dest_dir1)
convert_and_move_images(images_for_folder2, dest_dir2)
convert_and_move_images(images_for_folder3, dest_dir3)

print(f"Successfully moved 10,000 grayscale images to {dest_dir1}, {dest_dir2}, and {dest_dir3}.")
