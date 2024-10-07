"""
This scripts view the images in the csv file based on index
"""
import pandas as pd
import matplotlib.pyplot as plt

def view_image_from_csv(csv_file_path, image_index):
    """
    View a specific image from a CSV file containing grayscale images.
    
    Parameters:
    - csv_file_path: str - Path to the CSV file.
    - image_index: int - Index of the image to view.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get the image data
    image_data = df.iloc[image_index].values
    
    # Reshape the data to 48x48
    try:
        image = image_data.reshape(48, 48)
    except ValueError:
        print(f"Error reshaping image at index {image_index}. Data length: {len(image_data)}")
        return

    # Display the image
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)  # Adjust vmin and vmax for scaling
    plt.axis('off')  # Hide axes
    plt.show()

csv_path = r"path/to/your/csv"
view_image_from_csv(csv_path, 0)  # Csv path and index of image
