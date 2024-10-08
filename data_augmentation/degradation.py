"""
This scripts degrade the image in 3 different ways, and in 3 different levels:
- Cracks (Random white lines)
- Ink fade (Change brightness)
- Ink Stain (Apply a patch of rectangle blocking the character)
"""

import os
import cv2
import numpy as np
from skimage import util
import random

def degrade_images(input_folder, output_folder, degradation_level):
    """
    This is the main function that apply all the augmentation to all the images in three different levels.
    -Level 1: Somewhat degraded but still readable
    -Level 2: Partially degraded but still readable for some characters
    -Level 3: Severely degraded, very hard to read

    Parameters:
        -input_folder: Path to clean images (original images)
        -output_folder: Path to output images (degraded images)
        -degradation_level: Intensity of degradation

    """

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):  # Only process image files
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale input
            
            # Choose degradation parameters based on the level
            if degradation_level == 1:
                crack_intensity = 0.3
                fade_amount = 0.1
                stain_intensity = 0.25
                stain_size = 20
            elif degradation_level == 2:
                crack_intensity = 0.5
                fade_amount = 0.3
                stain_intensity = 0.3
                stain_size = 25
            elif degradation_level == 3:
                crack_intensity = 0.6
                fade_amount = 0.3
                stain_intensity = 0.35
                stain_size = 30

            # Apply degradation
            degraded_img = apply_cracks_and_fade(img, crack_intensity=crack_intensity, fade_amount=fade_amount)
            degraded_img = apply_stain(degraded_img, stain_intensity=stain_intensity, stain_size=stain_size)
            
            # Save the degraded image to the output folder
            output_img_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_img_path, degraded_img)

def apply_cracks_and_fade(image, crack_intensity, fade_amount):
    """
    This function tries to simulate cracks and ink fade on the dataset.

    Parameters:
        -image: Individual input of a clean image
        -crack_intensity: Number of cracks
        -fade_amount: Amount of darkness apply
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply fading by reducing brightness, keeping the image in grayscale
    faded_image = cv2.convertScaleAbs(image, alpha=(1 - fade_amount), beta=0)

    # Generate a random crack pattern (mask), ensuring it is grayscale (single channel)
    crack_mask = create_crack_mask(faded_image.shape, intensity=crack_intensity)

    # Ensure crack_mask is in uint8 format (0-255) and single channel
    if crack_mask.max() <= 1:
        crack_mask = (crack_mask * 255).astype(np.uint8)

    # Convert faded image to uint8 if it's not already, and ensure it's single-channel
    faded_image = faded_image.astype(np.uint8)

    # Blend the crack mask with the faded image (both grayscale)
    combined_image = cv2.addWeighted(faded_image, 1, crack_mask, 0.5, 0)

    return combined_image

def create_crack_mask(image_shape, intensity):
    """
    This function generates a crack pattern.

    Parameters:
        -image_shape: Size of image (48, 48)
        -intensity: Amount of cracks apply
    """
    # Create a blank mask with random cracks
    mask = np.zeros(image_shape, dtype=np.float32)
    
    # Add random cracks by drawing lines on the mask
    num_cracks = int(10 * intensity)  # More intensity = more cracks
    for _ in range(num_cracks):
        x1, y1 = random.randint(0, image_shape[1] - 1), random.randint(0, image_shape[0] - 1)
        x2, y2 = random.randint(0, image_shape[1] - 1), random.randint(0, image_shape[0] - 1)
        cv2.line(mask, (x1, y1), (x2, y2), color=1, thickness=random.randint(1, 3))

    # Add some random noise to simulate smaller cracks
    mask = util.random_noise(mask, mode='salt', amount=intensity / 10)

    return mask

def apply_stain(image, stain_intensity, stain_size):
    """
    This function simulates the effect of stain on an image by applying randomly sized and placed stain as a rectangular shape.

    Parameters:
        -image: Individual input of a clean image
        -stain_intensity: How strong or dark the stain is
        -stain_size: Size of the stain
    """
    # Create a copy of the image to apply stains
    stained_image = image.copy()

    # Randomly choose the number of stains to apply
    num_stains = random.randint(1, 5)

    for _ in range(num_stains):
        # Randomly select the position for the stain
        x = random.randint(0, stained_image.shape[1] - stain_size)
        y = random.randint(0, stained_image.shape[0] - stain_size)

        # Create a random stain patch
        stain_patch = np.full((stain_size, stain_size), fill_value=random.randint(180, 255), dtype=np.uint8)

        # Blend the stain patch with the stained_image
        stained_image[y:y + stain_size, x:x + stain_size] = cv2.addWeighted(
            stained_image[y:y + stain_size, x:x + stain_size],
            1 - stain_intensity,
            stain_patch,
            stain_intensity,
            0
        )

    return stained_image

input_folder = r'path/to/clean/images/folder'
output_folder = r'path/to/output/folder'
degradation_level = 1  
degrade_images(input_folder, output_folder, degradation_level)
