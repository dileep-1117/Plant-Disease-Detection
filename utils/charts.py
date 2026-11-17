"""
Interactive Charting Module
============================
Creates Altair charts styled to the Dark Green Agricultural Theme:
- Main background: #0B1712
- Card background: #163426
- Primary green: #22C55E
- Light green: #86EFAC
- Text: #F8FAFC, Secondary: #CBD5E1
- Grid/Borders: #28543D
"""

from typing import Any, Dict, List
import altair as alt
import pandas as pd


def apply_dark_theme(chart: alt.Chart) -> alt.Chart:
    """Apply consistent dark-green theme formatting to an Altair chart."""
    return (
        chart.configure_view(
            strokeWidth=0,
            fill="#163426",
        )
        .configure_axis(
            domainColor="#28543D",
            gridColor="#1F4232",
            labelColor="#CBD5E1",
            titleColor="#86EFAC",
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight="bold",
        )
        .configure_title(
            color="#F8FAFC",
            fontSize=14,
            fontWeight="bold",
        )
    )


def create_top5_chart(top_5_data: List[Dict[str, Any]]) -> alt.Chart:
    """
    Generate horizontal bar chart for top 5 model predictions.
    """
    records = []
    for item in top_5_data:
        records.append({
            "Condition": item.get("display_label", f"{item.get('plant_name')} – {item.get('disease_name')}"),
            "Plant": item.get("plant_name", ""),
            "Disease": item.get("disease_name", ""),
            "Confidence (%)": round(float(item.get("confidence_score", 0.0)), 2),
            "Label": f"{float(item.get('confidence_score', 0.0)):.1f}%",
        })

    df = pd.DataFrame(records)

    base = alt.Chart(df).encode(
        x=alt.X(
            "Confidence (%):Q",
            title="Model Softmax Confidence (%)",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(grid=True),
        ),
        y=alt.Y(
            "Condition:N",
            sort="-x",
            title="",
            axis=alt.Axis(labelLimit=300),
        ),
        tooltip=[
            alt.Tooltip("Plant:N", title="Plant"),
            alt.Tooltip("Disease:N", title="Condition"),
            alt.Tooltip("Confidence (%):Q", format=".2f", title="Confidence (%)"),
        ],
    )

    bars = base.mark_bar(
        cornerRadiusTopRight=6,
        cornerRadiusBottomRight=6,
        color="#22C55E",
    )

    text = base.mark_text(
        align="left",
        baseline="middle",
        dx=8,
        color="#F8FAFC",
        fontWeight="bold",
        fontSize=12,
    ).encode(text="Label:N")

    chart = (bars + text).properties(height=230, width="container")
    return apply_dark_theme(chart)


def create_all38_chart(all_38_data: List[Dict[str, Any]]) -> alt.Chart:
    """
    Generate complete horizontal bar chart for all 38 classes sorted descending.
    """
    records = []
    for item in all_38_data:
        records.append({
            "Rank": item.get("rank", 0),
            "Condition": item.get("display_label", f"{item.get('plant_name')} – {item.get('disease_name')}"),
            "Plant": item.get("plant_name", ""),
            "Disease": item.get("disease_name", ""),
            "Confidence (%)": round(float(item.get("confidence_score", 0.0)), 3),
            "Label": f"{float(item.get('confidence_score', 0.0)):.2f}%",
        })

    df = pd.DataFrame(records)
    max_val = max(100.0, float(df["Confidence (%)"].max()) + 5.0) if not df.empty else 100.0

    base = alt.Chart(df).encode(
        x=alt.X(
            "Confidence (%):Q",
            title="Confidence Score (%)",
            scale=alt.Scale(domain=[0, max_val]),
            axis=alt.Axis(grid=True),
        ),
        y=alt.Y(
            "Condition:N",
            sort="-x",
            title="",
            axis=alt.Axis(labelLimit=350),
        ),
        tooltip=[
            alt.Tooltip("Rank:Q", title="Rank"),
            alt.Tooltip("Plant:N", title="Plant"),
            alt.Tooltip("Disease:N", title="Disease"),
            alt.Tooltip("Confidence (%):Q", format=".3f", title="Confidence (%)"),
        ],
    )

    bars = base.mark_bar(
        cornerRadiusTopRight=4,
        cornerRadiusBottomRight=4,
        color="#10B981",
    )

    text = base.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="#86EFAC",
        fontSize=11,
        fontWeight="bold",
    ).encode(text="Label:N")

    chart = (bars + text).properties(height=880, width="container")
    return apply_dark_theme(chart)


def create_plant_wise_chart(plant_scores_data: List[Dict[str, Any]]) -> alt.Chart:
    """
    Generate horizontal bar chart for plant-wise aggregated probabilities across all 14 crop species.
    """
    records = []
    for item in plant_scores_data:
        records.append({
            "Plant": item.get("plant_name", ""),
            "Aggregated Score (%)": round(float(item.get("score_pct", 0.0)), 2),
            "Label": item.get("formatted_score", f"{float(item.get('score_pct', 0.0)):.1f}%"),
        })

    df = pd.DataFrame(records)

    base = alt.Chart(df).encode(
        x=alt.X(
            "Aggregated Score (%):Q",
            title="Aggregated Plant Probability (%)",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(grid=True),
        ),
        y=alt.Y(
            "Plant:N",
            sort="-x",
            title="",
            axis=alt.Axis(labelLimit=250),
        ),
        tooltip=[
            alt.Tooltip("Plant:N", title="Plant Species"),
            alt.Tooltip("Aggregated Score (%):Q", format=".2f", title="Aggregated Score (%)"),
        ],
    )

    bars = base.mark_bar(
        cornerRadiusTopRight=5,
        cornerRadiusBottomRight=5,
        color="#22C55E",
    )

    text = base.mark_text(
        align="left",
        baseline="middle",
        dx=6,
        color="#86EFAC",
        fontWeight="bold",
        fontSize=11,
    ).encode(text="Label:N")

    chart = (bars + text).properties(height=380, width="container")
    return apply_dark_theme(chart)


def create_history_chart(history_data: List[Dict[str, Any]]) -> alt.Chart:
    """
    Generate bar chart displaying chronological prediction history entries.
    """
    if not history_data:
        return None

    records = []
    for idx, item in enumerate(history_data):
        label_trunc = item.get("filename", f"Leaf-{idx+1}")[:14]
        records.append({
            "Entry": f"#{idx+1} {label_trunc} ({item.get('plant_name')})",
            "Timestamp": item.get("timestamp", ""),
            "Filename": item.get("filename", ""),
            "Plant": item.get("plant_name", ""),
            "Disease": item.get("disease_name", ""),
            "Confidence (%)": round(float(item.get("confidence_score", 0.0)), 2),
            "Reliability": item.get("reliability", "Reliable"),
            "Health": item.get("status", "Diseased"),
        })

    df = pd.DataFrame(records)

    base = alt.Chart(df).encode(
        x=alt.X(
            "Confidence (%):Q",
            title="AI Model Confidence (%)",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(grid=True),
        ),
        y=alt.Y(
            "Entry:N",
            sort=None,
            title="Diagnostic Session History (Chronological)",
            axis=alt.Axis(labelLimit=300),
        ),
        tooltip=[
            alt.Tooltip("Timestamp:N", title="Timestamp"),
            alt.Tooltip("Filename:N", title="Image File"),
            alt.Tooltip("Plant:N", title="Plant"),
            alt.Tooltip("Disease:N", title="Diagnosis"),
            alt.Tooltip("Confidence (%):Q", format=".2f", title="Confidence (%)"),
            alt.Tooltip("Reliability:N", title="Reliability"),
        ],
    )

    bars = base.mark_bar(
        cornerRadiusTopRight=5,
        cornerRadiusBottomRight=5,
    ).encode(
        color=alt.condition(
            alt.datum.Reliability == "Reliable",
            alt.value("#22C55E"),  # Primary Green
            alt.value("#EAB308"),  # Amber for uncertain
        )
    )

    text = base.mark_text(
        align="left",
        baseline="middle",
        dx=6,
        color="#F8FAFC",
        fontWeight="bold",
        fontSize=11,
    ).encode(text=alt.Text("Confidence (%):Q", format=".1f"))

    height = max(180, len(df) * 36)
    chart = (bars + text).properties(height=height, width="container")
    return apply_dark_theme(chart)
