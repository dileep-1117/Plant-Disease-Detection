"""
Prediction Engine Module
========================
Executes independent, model-driven inference for plant disease classification.

Guarantees:
- Every image is processed independently without state leakage or cached predictions.
- No hardcoded predictions or assumptions.
- Extracts plant name and disease name from the actual predicted class.
- Computes actual softmax prediction score multiplied by 100.
- Returns top-1 and top-5 predictions with confidence percentages and formatted labels.
- Validates that the prediction vector contains exactly 38 classes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from PIL import Image
import io

from model_loader import load_class_names, load_trained_model
from preprocessing import preprocess_image


def format_class_name(raw_name: str) -> Dict[str, Any]:
    """
    Format raw PlantVillage class names into human-readable components:
    - Plant / Leaf Name (e.g. 'Tomato', 'Apple', 'Corn', 'Pepper (Bell)')
    - Disease Name / Condition (e.g. 'Early Blight', 'Common Rust', 'Healthy')
    - Display Name (e.g. 'Tomato - Early Blight')
    - is_healthy flag

    Example class mappings:
        Tomato___Early_blight -> Plant: 'Tomato', Disease: 'Early Blight'
        Apple___Apple_scab    -> Plant: 'Apple', Disease: 'Apple Scab'
        Corn___Common_rust    -> Plant: 'Corn', Disease: 'Common Rust'
        Apple___healthy       -> Plant: 'Apple', Disease: 'Healthy'
    """
    parts = raw_name.split("___")
    plant_raw = parts[0]
    condition_raw = parts[1] if len(parts) > 1 else ""

    # Clean plant name
    if plant_raw.lower() in ["pepper,_bell", "pepper, bell", "pepper_bell"]:
        plant_display = "Pepper (Bell)"
    else:
        plant_display = plant_raw.replace("_", " ").strip().title()

    # Clean disease/condition name
    is_healthy = "healthy" in condition_raw.lower()
    if is_healthy:
        disease_display = "Healthy"
    else:
        # Standardize known abbreviations and parenthetical phrases
        disease_display = condition_raw.replace("_", " ").strip()
        # Clean double spaces
        disease_display = " ".join(disease_display.split())
        # Clean specific common dataset names
        if "Cercospora" in disease_display and "Gray" in disease_display:
            disease_display = "Cercospora Leaf Spot (Gray Leaf Spot)"
        elif "Two-spotted" in disease_display or "Two-Spotted" in disease_display:
            disease_display = "Spider Mites (Two-Spotted Spider Mite)"
        elif "Haunglongbing" in disease_display:
            disease_display = "Huanglongbing (Citrus Greening)"
        else:
            disease_display = disease_display.title()

    display_name = f"{plant_display} - {disease_display}"

    return {
        "raw": raw_name,
        "plant": plant_display,
        "plant_name": plant_display,
        "condition": disease_display,
        "disease_name": disease_display,
        "display_name": display_name,
        "is_healthy": is_healthy,
    }


def predict_disease(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, Any],
    model: Optional[Any] = None,
    class_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Execute end-to-end prediction on a single image.

    Steps:
        1. Preprocess the image independently matching training configuration.
        2. Run forward pass with training=False (disabling data augmentation / dropout).
        3. Extract raw softmax probability vector of length 38.
        4. Find highest prediction and rank top_k predictions.
        5. Map index to the exact class name from class_names.json.
        6. Extract plant name, disease name, and model confidence percentage (prob * 100).

    Args:
        image_input: Path, bytes, BytesIO, PIL Image, or UploadedFile.
        model: Optional preloaded Keras model. If None, loads via model_loader.
        class_names: Optional preloaded class list. If None, loads via model_loader.
        top_k: Number of highest ranking predictions to return (default 5).

    Returns:
        Structured prediction dictionary with top prediction, top_k list,
        image metadata, raw probabilities, and confidence percentages.
    """
    # Step 1: Preprocess image independently
    image_tensor, pil_image, image_hash = preprocess_image(image_input)

    # Step 2: Ensure model and class names are available
    if model is None:
        model = load_trained_model()
    if class_names is None:
        class_names = load_class_names()

    num_classes = len(class_names)
    if num_classes != 38:
        raise ValueError(f"Expected 38 classes, got {num_classes}")

    # Step 3: Run model inference (training=False ensures deterministic evaluation)
    raw_output = model(image_tensor, training=False)
    if hasattr(raw_output, "numpy"):
        raw_probs = raw_output.numpy()[0]
    else:
        raw_probs = np.array(raw_output)[0]

    # Validate output vector
    if raw_probs.shape[0] != 38:
        raise ValueError(
            f"Prediction output dimension mismatch: expected 38, got {raw_probs.shape[0]}"
        )

    # Step 4: Top predictions
    top_indices = np.argsort(raw_probs)[::-1][:top_k]

    top_predictions_list: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        idx_int = int(idx)
        raw_name = class_names[idx_int]
        prob = float(raw_probs[idx_int])
        score_100 = prob * 100.0
        info = format_class_name(raw_name)

        top_predictions_list.append(
            {
                "rank": rank,
                "class_index": idx_int,
                "class_name": raw_name,
                "plant": info["plant_name"],
                "plant_name": info["plant_name"],
                "condition": info["disease_name"],
                "disease_name": info["disease_name"],
                "display_name": info["display_name"],
                "is_healthy": info["is_healthy"],
                "confidence": prob,
                "confidence_score": score_100,
                "confidence_pct": f"{score_100:.1f}%",
                "confidence_pct_precise": f"{score_100:.2f}%",
            }
        )

    best_prediction = top_predictions_list[0]

    return {
        "top_prediction": best_prediction,
        "top_5": top_predictions_list,
        "image_hash": image_hash,
        "image_size": pil_image.size,
        "raw_probabilities": raw_probs,
        "num_classes": num_classes,
        "tensor_shape": list(image_tensor.shape),
    }
