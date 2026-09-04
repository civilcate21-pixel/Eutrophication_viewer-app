"""Carlson TSI trophic-state classification constants and helpers.

Ported directly from the source notebook (cells 7-ish) - logic unchanged.
"""
import pandas as pd

TROPHIC_CLASSES = [
    {"label": "Oligotrophic",             "range": "TSI < 30", "color": "#2196F3",
     "short": "Low nutrient, clear water"},
    {"label": "Oligotrophic-Mesotrophic", "range": "30-40",    "color": "#4CAF50",
     "short": "Low to moderate nutrient"},
    {"label": "Mesotrophic",              "range": "40-50",    "color": "#8BC34A",
     "short": "Moderate nutrient loading"},
    {"label": "Mesotrophic-Eutrophic",    "range": "50-60",    "color": "#FFC107",
     "short": "Elevated nutrients"},
    {"label": "Eutrophic",                "range": "60-70",    "color": "#FF5722",
     "short": "High nutrients, bloom risk"},
    {"label": "Hypereutrophic",           "range": "TSI >= 70", "color": "#B71C1C",
     "short": "Severe, immediate action"},
]

PALETTE_TSI      = [tc["color"] for tc in TROPHIC_CLASSES]
PALETTE_CHLA     = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff8800", "#ff0000"]
PALETTE_NDCI     = ["#1a0066", "#1400ff", "#00bbff", "#00ff44", "#ffee00", "#ff0000"]
PALETTE_MNDWI    = ["#ffffff", "#b3e5fc", "#4fc3f7", "#0288d1", "#01579b", "#002244"]
PALETTE_FAI      = ["#000080", "#0000ff", "#00ffff", "#7fff00", "#ffff00", "#ff0000"]
PALETTE_CIREDEGE = ["#000000", "#004400", "#00ff00", "#ffff00", "#ff8800", "#ff0000"]

TROPHIC_MARKER = [
    {"label": tc["label"], "range": tc["range"], "color": "#00000000", "short": tc["short"]}
    for tc in TROPHIC_CLASSES
]


def tsi_to_class_id(tsi: float) -> int:
    if tsi < 30:
        return 0
    elif tsi < 40:
        return 1
    elif tsi < 50:
        return 2
    elif tsi < 60:
        return 3
    elif tsi < 70:
        return 4
    else:
        return 5


def classify_tsi(tsi: float) -> dict:
    cid = tsi_to_class_id(tsi)
    tc = TROPHIC_CLASSES[cid]
    recs = [
        "Excellent quality - maintain current conditions",
        "Good quality - monitor regularly",
        "Moderate quality - seasonal monitoring advised",
        "Declining quality - investigate nutrient sources",
        "Poor quality - implement remediation measures",
        "Critical - immediate intervention required",
    ]
    details = [
        "Low nutrient, crystal-clear water",
        "Low to moderate nutrient",
        "Moderate nutrient loading",
        "Elevated nutrients, reduced clarity",
        "High nutrients, algal bloom risk",
        "Excessive nutrients, heavy algal blooms",
    ]
    return {
        "TSI_Score": round(tsi, 2),
        "Trophic_Status": tc["label"],
        "Detail": details[cid],
        "Recommendation": recs[cid],
        "Color": tc["color"],
    }


def overall_trophic(result_df: pd.DataFrame) -> dict:
    return classify_tsi(round(result_df["TSI_Score"].mean(), 2))
