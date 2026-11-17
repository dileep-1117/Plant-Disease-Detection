"""
Improved Prediction Engine Module (Two-Stage Hierarchical Inference)
=====================================================================
Implements a scientifically grounded Two-Stage prediction system for the
38-class Plant Disease Detection model:

Stage 1: Plant Species Identification
    - Dynamically maps the 38 PlantVillage classes to their 14 parent plant species.
    - Computes the marginal probability of each plant species by summing the
      softmax probabilities across all conditions for that crop:
      P(Plant = S_k | X) = sum_{c_i in Classes(S_k)} P(c_i | X)
    - Selects the most likely plant species without relying on filenames.

Stage 2: Plant-Conditioned Disease Identification
    - Restricts disease candidates strictly to the identified plant species.
    - Computes conditional disease probability within that plant's subspace:
      P(Disease c_i | Plant S_k, X) = P(c_i | X) / sum_{c_j in Classes(S_k)} P(c_j | X)
    - Guarantees that cross-species disease hallucination is mathematically impossible.

Reliability & Uncertainty Evaluation:
    - Analyzes raw confidence and inter-class ambiguity margins.
    - If confidence is below threshold or the margin between candidates is ambiguous,
      flags the prediction as UNRELIABLE and surfaces the explicit advisory:
      "Unable to identify this leaf reliably. Please upload a clearer image."

Graph & Dashboard Payloads:
    - Top 5 Predictions with 'Plant – Disease' display labels and confidence scores.
    - Full All-38 Class Predictions ranked descending by confidence.
    - Plant-Wise Aggregated Scores for all 14 crop species.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from PIL import Image
import io

from model_loader import load_class_names, load_trained_model
from preprocessing import preprocess_image


def get_plant_species_catalog(class_names: List[str]) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """
    Build two-way mappings between the 38 classes and their parent plant species.
    
    Returns:
        plant_to_classes: Dict[plant_name, list_of_class_indices]
        class_to_plant: Dict[class_index, plant_name]
    """
    plant_to_classes: Dict[str, List[int]] = {}
    class_to_plant: Dict[int, str] = {}

    for idx, c in enumerate(class_names):
        plant_raw = c.split("___")[0].replace("_", " ").replace(",", "").strip()
        # Standardize readable crop names
        if plant_raw.lower() in ["pepper bell", "pepper, bell"]:
            plant_name = "Pepper (Bell)"
        elif "cherry" in plant_raw.lower():
            plant_name = "Cherry"
        elif "corn" in plant_raw.lower():
            plant_name = "Corn"
        else:
            plant_name = plant_raw.title()

        if plant_name not in plant_to_classes:
            plant_to_classes[plant_name] = []
        plant_to_classes[plant_name].append(idx)
        class_to_plant[idx] = plant_name

    return plant_to_classes, class_to_plant


def clean_disease_name(raw_class: str) -> Tuple[str, bool]:
    """Extract clean disease/condition label and health status."""
    parts = raw_class.split("___")
    condition_raw = parts[1] if len(parts) > 1 else raw_class

    is_healthy = "healthy" in condition_raw.lower()
    if is_healthy:
        return "Healthy", True

    disease = condition_raw.replace("_", " ").strip()
    disease = " ".join(disease.split())
    if "Cercospora" in disease and "Gray" in disease:
        disease = "Cercospora Leaf Spot (Gray Leaf Spot)"
    elif "Two-spotted" in disease or "Two-Spotted" in disease:
        disease = "Spider Mites (Two-Spotted Spider Mite)"
    elif "Haunglongbing" in disease:
        disease = "Huanglongbing (Citrus Greening)"
    else:
        disease = disease.title()

    return disease, False


def predict_two_stage(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, Any],
    model: Optional[Any] = None,
    class_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.40,
    ambiguity_threshold: float = 0.08,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Execute Two-Stage Hierarchical Inference with Uncertainty Detection and
    full dashboard graph payloads.
    """
    # Preprocess image independently
    image_tensor, pil_image, image_hash = preprocess_image(image_input)

    if model is None:
        model = load_trained_model()
    if class_names is None:
        class_names = load_class_names()

    # Forward pass with training=False
    raw_output = model(image_tensor, training=False)
    if hasattr(raw_output, "numpy"):
        raw_probs = raw_output.numpy()[0]
    else:
        raw_probs = np.array(raw_output)[0]

    plant_to_classes, class_to_plant = get_plant_species_catalog(class_names)

    # -------------------------------------------------------------
    # STAGE 1: Plant Species Identification
    # Marginalize softmax probabilities by crop category
    # -------------------------------------------------------------
    plant_probs: Dict[str, float] = {}
    for plant, idxs in plant_to_classes.items():
        plant_probs[plant] = float(np.sum(raw_probs[idxs]))

    sorted_plants = sorted(plant_probs.items(), key=lambda x: x[1], reverse=True)
    top_plant, top_plant_prob = sorted_plants[0]
    second_plant, second_plant_prob = sorted_plants[1] if len(sorted_plants) > 1 else ("", 0.0)

    # -------------------------------------------------------------
    # STAGE 2: Disease Identification (Strictly within identified plant)
    # -------------------------------------------------------------
    candidate_indices = plant_to_classes[top_plant]
    candidate_scores = {idx: float(raw_probs[idx]) for idx in candidate_indices}
    sum_candidates = sum(candidate_scores.values())

    if sum_candidates > 1e-7:
        cond_scores = {idx: p / sum_candidates for idx, p in candidate_scores.items()}
    else:
        cond_scores = candidate_scores

    sorted_cond_diseases = sorted(cond_scores.items(), key=lambda x: x[1], reverse=True)
    best_disease_idx, best_cond_conf = sorted_cond_diseases[0]
    second_disease_idx, second_cond_conf = sorted_cond_diseases[1] if len(sorted_cond_diseases) > 1 else (None, 0.0)

    predicted_raw_class = class_names[best_disease_idx]
    clean_disease, is_healthy = clean_disease_name(predicted_raw_class)

    # Raw model probability for this class
    raw_model_prob = float(raw_probs[best_disease_idx])
    confidence_score = raw_model_prob * 100.0

    # -------------------------------------------------------------
    # UNCERTAINTY & AMBIGUITY EVALUATION
    # -------------------------------------------------------------
    disease_margin = best_cond_conf - second_cond_conf
    plant_margin = top_plant_prob - second_plant_prob

    is_reliable = True
    reliability_status = "RELIABLE"
    reliability_message = ""

    if raw_model_prob < confidence_threshold or top_plant_prob < 0.35:
        is_reliable = False
        reliability_status = "UNCERTAIN_LOW_CONFIDENCE"
        reliability_message = "Unable to identify this leaf reliably. Please upload a clearer image."
    elif disease_margin < ambiguity_threshold or plant_margin < 0.05:
        is_reliable = False
        reliability_status = "UNCERTAIN_AMBIGUOUS"
        reliability_message = "Unable to identify this leaf reliably. Please upload a clearer image."

    # -------------------------------------------------------------
    # TOP 5 GLOBAL PREDICTIONS
    # -------------------------------------------------------------
    top_global_indices = np.argsort(raw_probs)[::-1][:top_k]
    top_5_list: List[Dict[str, Any]] = []

    for rank, idx in enumerate(top_global_indices, start=1):
        idx_int = int(idx)
        raw_name = class_names[idx_int]
        p_name = class_to_plant[idx_int]
        d_name, d_healthy = clean_disease_name(raw_name)
        prob = float(raw_probs[idx_int])

        top_5_list.append({
            "rank": rank,
            "class_index": idx_int,
            "class_name": raw_name,
            "plant_name": p_name,
            "disease_name": d_name,
            "display_label": f"{p_name} – {d_name}",
            "is_healthy": d_healthy,
            "confidence": prob,
            "confidence_score": prob * 100.0,
            "confidence_pct": f"{prob * 100.0:.1f}%",
        })

    # -------------------------------------------------------------
    # ALL 38 CLASSES RANKED DESCENDING (For Complete Distribution Chart)
    # -------------------------------------------------------------
    all_38_indices = np.argsort(raw_probs)[::-1]
    all_38_list: List[Dict[str, Any]] = []

    for rank, idx in enumerate(all_38_indices, start=1):
        idx_int = int(idx)
        raw_name = class_names[idx_int]
        p_name = class_to_plant[idx_int]
        d_name, d_healthy = clean_disease_name(raw_name)
        prob = float(raw_probs[idx_int])

        all_38_list.append({
            "rank": rank,
            "class_index": idx_int,
            "class_name": raw_name,
            "plant_name": p_name,
            "disease_name": d_name,
            "display_label": f"{p_name} – {d_name}",
            "is_healthy": d_healthy,
            "confidence": prob,
            "confidence_score": prob * 100.0,
            "confidence_pct": f"{prob * 100.0:.2f}%",
        })

    # -------------------------------------------------------------
    # PLANT-WISE AGGREGATED SCORES
    # -------------------------------------------------------------
    plant_scores_list: List[Dict[str, Any]] = []
    for plant, prob in sorted_plants:
        plant_scores_list.append({
            "plant_name": plant,
            "probability": prob,
            "score_pct": round(prob * 100.0, 2),
            "formatted_score": f"{prob * 100.0:.1f}%"
        })

    return {
        "plant_name": top_plant,
        "disease_name": clean_disease,
        "display_label": f"{top_plant} – {clean_disease}",
        "class_name": predicted_raw_class,
        "class_index": best_disease_idx,
        "is_healthy": is_healthy,
        "confidence": raw_model_prob,
        "confidence_score": confidence_score,
        "confidence_pct": f"{confidence_score:.1f}%",
        "plant_confidence": top_plant_prob,
        "plant_confidence_pct": f"{top_plant_prob * 100.0:.1f}%",
        "conditional_disease_confidence": best_cond_conf,
        "conditional_disease_confidence_pct": f"{best_cond_conf * 100.0:.1f}%",
        "is_reliable": is_reliable,
        "reliability_status": reliability_status,
        "reliability_message": reliability_message,
        "plant_probabilities": sorted_plants,
        "plant_scores": plant_scores_list,
        "conditional_disease_probabilities": [
            (class_names[idx], conf) for idx, conf in sorted_cond_diseases
        ],
        "top_5": top_5_list,
        "all_38_predictions": all_38_list,
        "raw_probabilities": raw_probs,
        "image_hash": image_hash,
        "image_size": pil_image.size,
        "tensor_shape": list(image_tensor.shape),
    }
