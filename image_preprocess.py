import numpy as np
import cv2

# Function to load and preprocess the image to fit into the model
def load_and_preprocess_image(image_path, target_size=(48, 48)):
    """
    Load and preprocess image for inference.

    Parameters:
        -image_path: Path to each images
        -target_size=(48, 48): The size of the image

    Return:
        -image_reshaped: A 4D arrays of image for inference (number_of_image, width, height, channel)
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to the target size (model input size)
    image_resized = cv2.resize(image, target_size)
    
    # Normalize pixel values to range [0, 1]
    image_normalized = image_resized / 255.0
    
    # Reshape the image to match model input shape (batch_size, height, width, channels)
    # The batch size is 1 because we are predicting one image at a time
    image_reshaped = np.reshape(image_normalized, (1, target_size[0], target_size[1], 1))
    
    return image_reshaped