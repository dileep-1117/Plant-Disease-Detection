"""
Phase 5 Comprehensive Verification Test
=======================================
Tests the improved Two-Stage Prediction Engine using genuine labelled images
from 5 distinct plant species:
1. Apple
2. Tomato
3. Potato
4. Corn
5. Strawberry

Outputs:
- Actual plant vs Predicted plant
- Actual disease vs Predicted disease
- Confidence score and Reliability Status
- Correct / Incorrect evaluation
- Separate accuracy calculation for each plant species
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Import improved prediction engine
from model_loader import load_class_names, load_trained_model
from improved_prediction import predict_two_stage

SEPARATOR = "=" * 90
SUBSEP = "-" * 90

def run_phase5_tests():
    model = load_trained_model()
    class_names = load_class_names()

    target_species = ["Apple", "Tomato", "Potato", "Corn", "Strawberry"]
    test_dir = Path("dataset_test")

    if not test_dir.exists():
        print(f"[ERROR] Test directory {test_dir} not found.")
        return False

    all_test_images = sorted(list(test_dir.glob("*/*.*")))
    
    # Filter to only target species
    filtered_images = []
    for img_p in all_test_images:
        cls_folder = img_p.parent.name
        plant_raw = cls_folder.split("___")[0].replace("_", " ").strip()
        if plant_raw in target_species:
            filtered_images.append(img_p)

    print("\n" + SEPARATOR)
    print("PHASE 5: INDEPENDENT PLANT SPECIES & DISEASE VERIFICATION TEST")
    print(f"Target Species: {', '.join(target_species)}")
    print(f"Total Evaluated Test Images: {len(filtered_images)}")
    print(SEPARATOR)

    results_by_plant = defaultdict(list)
    overall_results = []

    print(f"{'#':<3} | {'Actual Plant':<11} | {'Pred Plant':<11} | {'Actual Disease':<26} | {'Pred Disease':<26} | {'Conf':<7} | {'Result':<7} | {'Reliability'}")
    print(SUBSEP)

    for i, img_path in enumerate(filtered_images, start=1):
        actual_class = img_path.parent.name
        actual_plant = actual_class.split("___")[0].replace("_", " ").strip()
        actual_disease_raw = actual_class.split("___")[1].replace("_", " ").strip()
        actual_disease = "Healthy" if "healthy" in actual_disease_raw.lower() else actual_disease_raw.title()
        if "Common Rust" in actual_disease:
            actual_disease = "Common Rust"

        res = predict_two_stage(img_path, model=model, class_names=class_names)

        pred_plant = res["plant_name"]
        pred_disease = res["disease_name"]
        conf_pct = res["confidence_pct"]
        is_reliable = res["is_reliable"]
        rel_status = res["reliability_status"]

        is_plant_match = (actual_plant.lower() == pred_plant.lower())
        is_disease_match = (actual_disease.lower() in pred_disease.lower() or pred_disease.lower() in actual_disease.lower())
        is_correct = is_plant_match and is_disease_match

        status_str = "CORRECT" if is_correct else "WRONG"
        rel_str = "Reliable" if is_reliable else "Uncertain"

        # Truncate strings for clean table alignment
        act_d_disp = actual_disease[:25]
        pred_d_disp = pred_disease[:25]

        print(f"{i:<3} | {actual_plant:<11} | {pred_plant:<11} | {act_d_disp:<26} | {pred_d_disp:<26} | {conf_pct:<7} | {status_str:<7} | {rel_str}")

        record = {
            "image": img_path.name,
            "actual_plant": actual_plant,
            "pred_plant": pred_plant,
            "actual_disease": actual_disease,
            "pred_disease": pred_disease,
            "confidence": res["confidence_score"],
            "is_plant_match": is_plant_match,
            "is_disease_match": is_disease_match,
            "is_correct": is_correct,
            "is_reliable": is_reliable,
            "rel_status": rel_status
        }
        results_by_plant[actual_plant].append(record)
        overall_results.append(record)

    print(SEPARATOR)
    print("ACCURACY REPORT BY PLANT SPECIES:")
    print(SEPARATOR)

    for plant in target_species:
        records = results_by_plant[plant]
        if not records:
            continue
        total = len(records)
        plant_correct = sum(1 for r in records if r["is_plant_match"])
        disease_correct = sum(1 for r in records if r["is_disease_match"])
        both_correct = sum(1 for r in records if r["is_correct"])
        reliable_count = sum(1 for r in records if r["is_reliable"])

        plant_acc = (plant_correct / total) * 100.0
        disease_acc = (disease_correct / total) * 100.0
        overall_acc = (both_correct / total) * 100.0

        print(f"Plant: {plant:<11} | Total: {total:2d} | Plant Accuracy: {plant_acc:5.1f}% | Disease Accuracy: {disease_acc:5.1f}% | Overall: {overall_acc:5.1f}% | Reliable: {reliable_count}/{total}")

    print(SEPARATOR)
    total_samples = len(overall_results)
    total_plant_corr = sum(1 for r in overall_results if r["is_plant_match"])
    total_corr = sum(1 for r in overall_results if r["is_correct"])
    print(f"OVERALL SUMMARY:")
    print(f"  Stage 1 (Plant Species Identification) Accuracy : {total_plant_corr}/{total_samples} ({total_plant_corr/total_samples*100:.2f}%)")
    print(f"  Stage 2 (Exact Disease Identification) Accuracy  : {total_corr}/{total_samples} ({total_corr/total_samples*100:.2f}%)")
    print(SEPARATOR + "\n")
    return True


if __name__ == "__main__":
    success = run_phase5_tests()
    sys.exit(0 if success else 1)
