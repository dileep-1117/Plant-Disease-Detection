"""
Model Loader Module
===================
Handles loading, validation, and metadata extraction for the trained
Plant Disease Detection Keras model and its associated class names.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import keras

# Default paths relative to project root
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "plant_disease_model.keras"
DEFAULT_CLASS_PATH = BASE_DIR / "class_names.json"
FALLBACK_CLASS_PATH = BASE_DIR / "class_name.json.json"

# Global memory cache for singleton model and class names
_CACHED_MODEL = None
_CACHED_CLASS_NAMES: Optional[List[str]] = None


def resolve_class_path(class_path: Optional[Path] = None) -> Path:
    """
    Resolve the path to the class names JSON file.
    Checks DEFAULT_CLASS_PATH first, then FALLBACK_CLASS_PATH.
    """
    if class_path is not None:
        p = Path(class_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified class names file not found: {p}")

    if DEFAULT_CLASS_PATH.exists():
        return DEFAULT_CLASS_PATH
    if FALLBACK_CLASS_PATH.exists():
        return FALLBACK_CLASS_PATH

    raise FileNotFoundError(
        f"Class names file not found. Checked: {DEFAULT_CLASS_PATH} and {FALLBACK_CLASS_PATH}"
    )


def load_class_names(class_path: Optional[Path] = None, force_reload: bool = False) -> List[str]:
    """
    Load and validate the 38 class names from JSON.
    Never reorders class names.

    Args:
        class_path: Optional custom path to class names JSON.
        force_reload: If True, bypasses in-memory cache.

    Returns:
        List of 38 class name strings.
    """
    global _CACHED_CLASS_NAMES
    if _CACHED_CLASS_NAMES is not None and not force_reload and class_path is None:
        return _CACHED_CLASS_NAMES

    resolved_path = resolve_class_path(class_path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list):
        raise TypeError(f"Expected class_names to be a list, got {type(class_names).__name__}")

    if len(class_names) != 38:
        raise ValueError(f"Expected exactly 38 classes, found {len(class_names)} in {resolved_path}")

    if class_path is None:
        _CACHED_CLASS_NAMES = class_names

    return class_names


def load_trained_model(model_path: Optional[Path] = None, force_reload: bool = False) -> keras.Model:
    """
    Load and validate the trained Keras CNN model.

    Args:
        model_path: Optional custom path to .keras model file.
        force_reload: If True, reloads model from disk.

    Returns:
        Loaded keras.Model instance.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and not force_reload and model_path is None:
        return _CACHED_MODEL

    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    # Load model without compiling since only inference is performed
    model = keras.models.load_model(str(path), compile=False)

    # Validate model input and output shapes
    expected_input_shape = (None, 96, 96, 3)
    if model.input_shape != expected_input_shape:
        raise ValueError(
            f"Model input shape mismatch: expected {expected_input_shape}, got {model.input_shape}"
        )

    expected_output_shape = (None, 38)
    if model.output_shape != expected_output_shape:
        raise ValueError(
            f"Model output shape mismatch: expected {expected_output_shape}, got {model.output_shape}"
        )

    if model_path is None:
        _CACHED_MODEL = model

    return model


def get_model_info(
    model: Optional[keras.Model] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Retrieve inspection metadata about the loaded model and classes.
    """
    if model is None:
        model = load_trained_model()
    if class_names is None:
        class_names = load_class_names()

    total_params = model.count_params()
    trainable_params = sum(
        keras.ops.size(w).item() if hasattr(keras.ops.size(w), "item") else int(keras.ops.size(w))
        for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    return {
        "model_path": str(DEFAULT_MODEL_PATH),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "num_classes": len(class_names),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "classes": class_names,
    }
