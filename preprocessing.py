"""
Preprocessing Module
====================
Provides strictly validated image loading and tensor transformation functions
matching the exact training pipeline for the Plant Disease model:
- Load image & auto-orient via EXIF
- Convert to RGB color mode
- Convert uint8 NumPy array to TensorFlow Tensor
- Resize to 96x96 with tf.image.resize
- Cast to tf.float32 and normalize [0, 1] by dividing by 255.0
- Expand dimensions to add batch dimension: shape (1, 96, 96, 3)
"""

import hashlib
import io
from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

IMG_HEIGHT = 96
IMG_WIDTH = 96
TARGET_SHAPE = (IMG_HEIGHT, IMG_WIDTH)


def compute_image_hash(image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> str:
    """
    Calculate a SHA-256 hash of the input image for verification and tracking.
    """
    if isinstance(image_input, (str, Path)):
        with open(image_input, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    elif isinstance(image_input, bytes):
        return hashlib.sha256(image_input).hexdigest()
    elif isinstance(image_input, io.BytesIO):
        pos = image_input.tell()
        image_input.seek(0)
        data = image_input.read()
        image_input.seek(pos)
        return hashlib.sha256(data).hexdigest()
    elif isinstance(image_input, Image.Image):
        # Hash raw pixel bytes
        return hashlib.sha256(image_input.tobytes()).hexdigest()
    elif hasattr(image_input, "getvalue"):  # Streamlit UploadedFile
        return hashlib.sha256(image_input.getvalue()).hexdigest()
    else:
        # Fallback to string representation
        return hashlib.sha256(str(image_input).encode("utf-8")).hexdigest()


def load_image_as_pil(image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> Image.Image:
    """
    Safely load an image from multiple possible sources and return an oriented RGB PIL Image.

    Supports:
        - File path (str or Path)
        - Raw bytes
        - io.BytesIO buffer
        - PIL Image instance
        - Streamlit UploadedFile (file-like object)
    """
    if isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        image = Image.open(path)
    elif isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "read"):
        # File-like object (e.g., io.BytesIO or Streamlit UploadedFile)
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        image = Image.open(image_input)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Correct orientation if EXIF orientation tag exists
    image = ImageOps.exif_transpose(image)

    # Ensure 3-channel RGB (handles RGBA, grayscale, CMYK, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def preprocess_image(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]
) -> Tuple[tf.Tensor, Image.Image, str]:
    """
    Preprocess an image strictly matching the original training configuration.

    Pipeline:
        1. Open image & convert to RGB
        2. image_array = np.array(image, dtype=np.uint8)
        3. image_tensor = tf.convert_to_tensor(image_array)
        4. image_tensor = tf.image.resize(image_tensor, (96, 96))
        5. image_tensor = tf.cast(image_tensor, tf.float32) / 255.0
        6. image_tensor = tf.expand_dims(image_tensor, axis=0)

    Returns:
        Tuple of:
            - image_tensor: tf.Tensor of shape (1, 96, 96, 3) with float32 values in [0, 1]
            - pil_image: The loaded PIL Image (RGB)
            - image_hash: SHA-256 hash string of the input image
    """
    image_hash = compute_image_hash(image_input)
    pil_image = load_image_as_pil(image_input)

    # Convert to uint8 numpy array
    image_array = np.array(pil_image, dtype=np.uint8)

    # Convert to TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image_array)

    # Resize to 96x96 using TensorFlow (bilinear interpolation)
    image_tensor = tf.image.resize(image_tensor, TARGET_SHAPE)

    # Normalize float32 to [0, 1]
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0

    # Expand dims to add batch dimension -> (1, 96, 96, 3)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    # Assert shape integrity
    if image_tensor.shape != (1, IMG_HEIGHT, IMG_WIDTH, 3):
        raise ValueError(
            f"Preprocessed tensor shape mismatch: expected (1, {IMG_HEIGHT}, {IMG_WIDTH}, 3), got {image_tensor.shape}"
        )

    return image_tensor, pil_image, image_hash
