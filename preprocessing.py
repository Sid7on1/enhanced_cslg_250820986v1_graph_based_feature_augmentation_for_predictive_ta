import logging
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from typing import List, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
TEMP_DIR = tempfile.mkdtemp()
INPUT_DIR = os.path.join(TEMP_DIR, "input")
OUTPUT_DIR = os.ephemeral_path("output")

# Exception classes
class PreprocessingError(Exception):
    """Custom exception class for preprocessing errors."""
    pass

# Main class with methods
class ImagePreprocessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.images = []
        self.labels = []

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_images(self) -> None:
        """
        Load images from the input directory and store their paths and labels.

        Raises:
            PreprocessingError: If the input directory does not exist or contains no images.
        """
        if not os.path.exists(self.input_dir):
            raise PreprocessingError(f"Input directory '{self.input_dir}' does not exist.")

        logger.info("Loading images from input directory...")
        for filename in os.listdir(self.input_dir):
            image_path = os.path.join(self.input_dir, filename)
            if os.path.isfile(image_path) and any(filename.endswith(ext) for ext in [".jpg", ".png", ".gif"]):
                self.images.append(image_path)
                self.labels.append(filename.split("_")[0])

        if not self.images:
            raise PreprocessingError("No valid images found in the input directory.")

        logger.info(f"Loaded {len(self.images)} images.")

    def resize_images(self, width: int, height: int) -> None:
        """
        Resize all loaded images to the specified width and height.

        Args:
            width (int): The target width for the images.
            height (int): The target height for the images.

        Raises:
            PreprocessingError: If any image fails to be resized.
        """
        logger.info(f"Resizing images to {width}x{height}...")
        for image_path in self.images:
            try:
                image = Image.open(image_path)
                image = image.resize((width, height))
                image.save(image_path)
            except Exception as e:
                raise PreprocessingError(f"Failed to resize image '{image_path}': {e}")

    def convert_to_grayscale(self) -> None:
        """Convert all loaded images to grayscale format."""
        logger.info("Converting images to grayscale...")
        for image_path in self.images:
            try:
                image = Image.open(image_path).convert("L")
                image.save(image_path)
            except Exception as e:
                logger.error(f"Failed to convert image '{image_path}' to grayscale: {e}")

    def normalize_images(self, mean: List[int], std: List[int]) -> None:
        """
        Normalize the pixel values of the images using the provided mean and standard deviation.

        Args:
            mean (List[int]): The mean pixel values for normalization.
            std (List[int]): The standard deviation values for normalization.
        """
        logger.info("Normalizing images...")
        for image_path in self.images:
            try:
                image = np.array(Image.open(image_path))
                image = image.astype(np.float32)
                image = (image - mean) / std
                Image.fromarray(image.astype(np.uint8)).save(image_path)
            except Exception as e:
                logger.error(f"Failed to normalize image '{image_path}': {e}")

    def augment_images(self, flip_horizontal: bool = False, flip_vertical: bool = False) -> None:
        """
        Apply data augmentation techniques to the loaded images.

        Args:
            flip_horizontal (bool): Whether to randomly flip images horizontally.
            flip_vertical (bool): Whether to randomly flip images vertically.
        """
        logger.info("Augmenting images...")
        for image_path in self.images:
            image = Image.open(image_path)

            # Horizontal flip
            if flip_horizontal and np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Vertical flip
            if flip_vertical and np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

            image.save(image_path)

    def save_images(self) -> None:
        """Save the preprocessed images to the output directory."""
        logger.info(f"Saving preprocessed images to '{self.output_dir}'...")
        os.makedirs(self.output_dir, exist_ok=True)

        for image_path, label in zip(self.images, self.labels):
            filename = f"{label}_{os.path.basename(image_path)}"
            output_path = os.path.join(self.output_dir, filename)
            shutil.copyfile(image_path, output_path)

        logger.info("Images saved successfully.")

    def preprocess(self, width: int, height: int, grayscale: bool = False,
                  mean: List[int] = None, std: List[int] = None,
                  flip_horizontal: bool = False, flip_vertical: bool = False) -> None:
        """
        Perform the complete preprocessing pipeline on the loaded images.

        Args:
            width (int): The target width for resizing images.
            height (int): The target height for resizing images.
            grayscale (bool): Whether to convert images to grayscale.
            mean (List[int], optional): Mean pixel values for normalization.
            std (List[int], optional): Standard deviation values for normalization.
            flip_horizontal (bool): Whether to randomly flip images horizontally.
            flip_vertical (bool): Whether to randomly flip images vertically.
        """
        self.load_images()
        self.resize_images(width, height)

        if grayscale:
            self.convert_to_grayscale()

        if mean is not None and std is not None:
            self.normalize_images(mean, std)

        self.augment_images(flip_horizontal, flip_vertical)
        self.save_images()

# Helper functions
def load_image(image_path: str) -> np.array:
    """
    Load an image from the given file path and return it as a numpy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: The loaded image as a numpy array.
    """
    try:
        image = Image.open(image_path)
        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to load image '{image_path}': {e}")
        return None

# Example usage
if __name__ == "__main__":
    preprocessor = ImagePreprocessor(INPUT_DIR, OUTPUT_DIR)
    preprocessor.preprocess(width=224, height=224, grayscale=True,
                           mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                           flip_horizontal=True, flip_vertical=True)