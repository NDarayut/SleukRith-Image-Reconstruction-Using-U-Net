from image_preprocess import load_and_preprocess_image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model 
model = load_model(r'complex_unet.h5')

# Path to the degraded images
image_path = r'\test_images\image_4.jpg'

# Load and preprocess the image
preprocessed_image = load_and_preprocess_image(image_path)

# Make a prediction using the trained model
prediction = model.predict(preprocessed_image)

# Reshape prediction back to the image dimensions for display (if necessary)
predicted_image = np.squeeze(prediction)

# Display the original and predicted images
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(original_image, cmap='gray')
plt.title('Degraded Image')
plt.axis('off')

# Predicted Image (Restored)
plt.subplot(1, 2, 2)
plt.imshow(predicted_image, cmap='gray')
plt.title('Restored image')
plt.axis('off')

plt.tight_layout()
plt.show()
