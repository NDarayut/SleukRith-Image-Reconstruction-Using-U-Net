"""
This scripts convert all the images into a 1D array and store it in CSV file format.
"""
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def images_to_csv(image_folder, csv_file, image_size=(48, 48)):
    """
    Convert all the images into an array of pixels value and stored as csv file format.

    Parameters:
        -image_folder: Path to your image folder
        -csv_file: Path and name of your to-be-created csv file
        -image_size=(48, 48): Size of the image
    """
    pixel_values = []

    # Iterate through each image in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):  # Add other formats if needed
            # Load image
            img_path = os.path.join(image_folder, filename)
            img = load_img(img_path, target_size=image_size, color_mode='grayscale')  # Use 'rgb' for color images
            
            # Convert image to array
            img_array = img_to_array(img)
            
            # Normalize pixel values to [0, 1]
            img_array /= 255.0
            
            # Flatten the image array and convert to a list
            img_flat = img_array.flatten().tolist()
            pixel_values.append(img_flat)

    # Create a DataFrame
    df = pd.DataFrame(pixel_values)

    # Save DataFrame to CSV
    df.to_csv(csv_file, index=False)

# Example usage
image_folder = r'your/image/folder/path'  # Folder containing images
csv_file = r'your/path/to/testset_degraded.csv'     # Desired CSV filename
images_to_csv(image_folder, csv_file)
