import os
import cv2
import numpy as np

# Directory paths
RAW_DATA_DIR = "data/dataset/raw/"
PROCESSED_DATA_DIR = "data/dataset/processed/"
IMG_SIZE = (50, 50)

def preprocess_images(input_dir, output_dir, img_size):
    """
    Preprocesses images by resizing and converting to grayscale.

    Args:
        input_dir (str): Path to the raw images directory.
        output_dir (str): Path to save processed images.
        img_size (tuple): Desired size for the images (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)

                if img is not None:
                    # Convert to grayscale
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Resize image
                    resized_img = cv2.resize(gray_img, img_size)

                    # Save processed image
                    relative_path = os.path.relpath(subdir, input_dir)
                    save_dir = os.path.join(output_dir, relative_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    save_path = os.path.join(save_dir, file)
                    cv2.imwrite(save_path, resized_img)

def load_dataset(data_dir):
    """
    Loads the dataset into arrays of images and labels.

    Args:
        data_dir (str): Path to the processed images directory.

    Returns:
        tuple: (images, labels) where images is a numpy array of image data
               and labels is a list of corresponding labels.
    """
    images = []
    labels = []

    for subdir, _, files in os.walk(data_dir):
        label = os.path.basename(subdir)
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                images.append(img.flatten())
                labels.append(label)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    print("Starting data preprocessing...")

    # Preprocess images
    preprocess_images(RAW_DATA_DIR, PROCESSED_DATA_DIR, IMG_SIZE)
    print(f"Images have been processed and saved to {PROCESSED_DATA_DIR}")

    # Load dataset for verification
    images, labels = load_dataset(PROCESSED_DATA_DIR)
    print(f"Loaded {len(images)} images with {len(set(labels))} unique labels.")
