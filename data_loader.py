import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = 'data'
IMAGE_DIR = 'images'
ANNOTATION_DIR = 'annotations'
BATCH_SIZE = 32
NUM_WORKERS = 4

# Data class for image metadata
@dataclass
class ImageMetadata:
    image_id: str
    image_path: str
    annotation_path: str
    width: int
    height: int

# Enum for data split
class DataSplit(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'

# Abstract base class for dataset
class DatasetBase(ABC):
    def __init__(self, data_dir: str, split: DataSplit):
        self.data_dir = data_dir
        self.split = split
        self.image_metadata = self.load_image_metadata()

    @abstractmethod
    def load_image_metadata(self) -> List[ImageMetadata]:
        pass

    def get_image(self, image_metadata: ImageMetadata) -> np.ndarray:
        image_path = image_metadata.image_path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_annotation(self, image_metadata: ImageMetadata) -> np.ndarray:
        annotation_path = image_metadata.annotation_path
        annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        return annotation

# Concrete dataset class for image data
class ImageDataset(DatasetBase):
    def load_image_metadata(self) -> List[ImageMetadata]:
        image_metadata = []
        for image_id in os.listdir(os.path.join(self.data_dir, IMAGE_DIR)):
            image_path = os.path.join(self.data_dir, IMAGE_DIR, image_id)
            annotation_path = os.path.join(self.data_dir, ANNOTATION_DIR, image_id)
            width, height = cv2.imread(image_path).shape[:2]
            image_metadata.append(ImageMetadata(image_id, image_path, annotation_path, width, height))
        return image_metadata

    def __len__(self):
        return len(self.image_metadata)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_metadata = self.image_metadata[index]
        image = self.get_image(image_metadata)
        annotation = self.get_annotation(image_metadata)
        return image, annotation

# Data loader class
class DataLoaderClass:
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

# Configuration class
@dataclass
class Config:
    data_dir: str
    split: DataSplit
    batch_size: int
    num_workers: int

# Main function
def main():
    config = Config(DATA_DIR, DataSplit.TRAIN, BATCH_SIZE, NUM_WORKERS)
    dataset = ImageDataset(config.data_dir, config.split)
    data_loader = DataLoaderClass(dataset, config.batch_size, config.num_workers)
    logger.info(f'Data loader created with batch size {config.batch_size} and num workers {config.num_workers}')
    for batch in data_loader:
        images, annotations = batch
        logger.info(f'Batch size: {images.shape[0]}')
        logger.info(f'Images shape: {images.shape}')
        logger.info(f'Annotations shape: {annotations.shape}')

if __name__ == '__main__':
    main()