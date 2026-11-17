"""
Verify App Rendering and Page Logic
"""
import sys
from pathlib import Path
import json

from model_loader import load_class_names, load_trained_model
from utils.disease_info import DISEASE_CATALOG, filter_catalog
from utils.history_manager import load_history, save_prediction, clear_history
from utils.charts import create_top5_chart, create_all38_chart, create_plant_wise_chart, create_history_chart
from improved_prediction import predict_two_stage

print("=" * 60)
print("Testing App Module Components...")
print("=" * 60)

# 1. Test model and class loading
model = load_trained_model()
class_names = load_class_names()
print(f"1. Model & Class Names loaded: {len(class_names)} classes.")

# 2. Test disease catalog
print(f"2. Disease Catalog entries: {len(DISEASE_CATALOG)}")
filtered = filter_catalog("Apple", "Scab")
print(f"   Filter 'Apple', 'Scab' returned: {len(filtered)} result(s).")
assert len(filtered) == 1, "Expected 1 result for Apple Scab"

# 3. Test Prediction & Chart generation
test_img = Path("data/samples/apple_scab_sample.jpg")
if test_img.exists():
    res = predict_two_stage(test_img, model=model, class_names=class_names)
    print(f"3. Two-Stage prediction executed: {res['display_label']} ({res['confidence_pct']})")
    
    chart_top5 = create_top5_chart(res["top_5"])
    assert chart_top5 is not None, "Top 5 chart failed"
    print("   Top 5 chart created successfully.")

    chart_all38 = create_all38_chart(res["all_38_predictions"])
    assert chart_all38 is not None, "All 38 chart failed"
    print("   All 38 chart created successfully.")

    chart_plants = create_plant_wise_chart(res["plant_scores"])
    assert chart_plants is not None, "Plant-wise chart failed"
    print("   Plant-wise chart created successfully.")

# 4. Test History Manager
hist_entry = save_prediction(
    filename="test_sample.jpg",
    plant_name="Apple",
    disease_name="Apple Scab",
    confidence_score=97.5,
    is_healthy=False,
    is_reliable=True,
    image_hash="abc123test",
)
history = load_history()
print(f"4. History manager verified: {len(history)} entries in history.")
chart_hist = create_history_chart(history)
assert chart_hist is not None, "History chart failed"
print("   History chart created successfully.")

print("=" * 60)
print("ALL APP MODULE COMPONENTS VERIFIED AND OPERATIONAL!")
print("=" * 60)
