"""
Prediction History Manager
===========================
Handles local session and file-based persistence for leaf diagnoses:
- Logs image filename, plant name, predicted disease, confidence score, and timestamp.
- Stores data in `data/prediction_history.json`.
- Provides retrieval, summary metrics, and safe clearing with confirmation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "prediction_history.json"


def ensure_history_file() -> Path:
    """Ensure data directory and history file exist."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
    return HISTORY_FILE


def load_history() -> List[Dict[str, Any]]:
    """Load logged predictions from local disk."""
    ensure_history_file()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_prediction(
    filename: str,
    plant_name: str,
    disease_name: str,
    confidence_score: float,
    is_healthy: bool,
    is_reliable: bool,
    image_hash: str = "",
) -> Dict[str, Any]:
    """
    Append a new prediction record to the local persistent history.
    """
    ensure_history_file()
    history = load_history()

    # Prevent immediate duplicate consecutive logs of the same image hash
    if history and image_hash and history[-1].get("image_hash") == image_hash:
        return history[-1]

    entry = {
        "id": len(history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_only": datetime.now().strftime("%H:%M:%S"),
        "date_only": datetime.now().strftime("%Y-%m-%d"),
        "filename": filename,
        "plant_name": plant_name,
        "disease_name": disease_name,
        "display_name": f"{plant_name} – {disease_name}",
        "confidence_score": round(confidence_score, 2),
        "confidence_pct": f"{confidence_score:.1f}%",
        "is_healthy": is_healthy,
        "is_reliable": is_reliable,
        "status": "Healthy" if is_healthy else "Diseased",
        "reliability": "Reliable" if is_reliable else "Uncertain",
        "image_hash": image_hash,
    }

    history.append(entry)

    # Keep latest 100 entries
    if len(history) > 100:
        history = history[-100:]

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write prediction history: {e}")

    return entry


def clear_history():
    """Clear all records from prediction history."""
    ensure_history_file()
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)


def get_history_summary(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for the prediction history."""
    if not history:
        return {
            "total": 0,
            "healthy_count": 0,
            "diseased_count": 0,
            "avg_confidence": 0.0,
            "reliable_count": 0,
            "unique_plants": 0,
        }

    total = len(history)
    healthy_count = sum(1 for h in history if h.get("is_healthy", False))
    diseased_count = total - healthy_count
    avg_conf = sum(h.get("confidence_score", 0.0) for h in history) / total
    reliable_count = sum(1 for h in history if h.get("is_reliable", True))
    unique_plants = len(set(h.get("plant_name", "") for h in history))

    return {
        "total": total,
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "avg_confidence": round(avg_conf, 1),
        "reliable_count": reliable_count,
        "unique_plants": unique_plants,
    }
