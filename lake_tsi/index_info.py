"""Descriptions of the spectral indices and the method, shown at the bottom of the app.

Formulas are written exactly as they are implemented in gee_utils.add_spectral_indices()
(Sentinel-2 band codes). If you change a formula there, change it here too.

`reference` is None where no citation has been added yet - the table shows "-" for those.
Fill them in with the papers you actually used before publishing.
"""

DATA_NOTE = (
    "**Data and method.** Sentinel-2 surface-reflectance images (COPERNICUS/S2_SR_HARMONIZED) "
    "are filtered by the chosen date range and cloud threshold, cloud-masked with the QA60 band, "
    "scaled to reflectance (÷10,000) and divided by π (an approximation of remote-sensing "
    "reflectance). The images are averaged per pixel into one composite and clipped to the lake "
    "boundary. The 8 bands (B2, B3, B4, B5, B6, B7, B8, B11) and the 9 indices below are the 17 "
    "inputs to the selected model, which predicts Carlson TSI for every pixel. "
    "Map layers are interpolated onto a display grid; the exact pixel values are in the CSV download."
)

INDEX_INFO = [
    {
        "Index": "MNDWI",
        "Formula (as implemented)": "(B4 − B11) / (B4 + B11)",
        "What it indicates": "Water vs. land contrast (open-water extent)",
        "Map layer": "Yes",
        "Reference": "Adapted from Xu (2006); implemented here with Red (B4) and SWIR (B11)",
    },
    {
        "Index": "NDCI",
        "Formula (as implemented)": "(B6 − B5) / (B6 + B5)",
        "What it indicates": "Chlorophyll-a proxy from red-edge bands",
        "Map layer": "Yes",
        "Reference": "Adapted from Mishra & Mishra (2012); implemented here with B6 and B5",
    },
    {
        "Index": "CI Red-Edge",
        "Formula (as implemented)": "(B8 − B5) / (B8 + B5)",
        "What it indicates": "Phytoplankton / chlorophyll proxy (NIR vs. red-edge 1)",
        "Map layer": "Yes",
        "Reference": None,
    },
    {
        "Index": "Hyph",
        "Formula (as implemented)": "(B4 − B5) / (B4 + B5)",
        "What it indicates": "Red vs. red-edge 1 contrast (model input)",
        "Map layer": "No",
        "Reference": None,
    },
    {
        "Index": "FAI",
        "Formula (as implemented)": "B8 − [B4 + (B11 − B4) × (842 − 665)/(1610 − 665)]",
        "What it indicates": "Floating algae / surface scum",
        "Map layer": "Yes",
        "Reference": "Hu (2009)",
    },
    {
        "Index": "AFAI",
        "Formula (as implemented)": "B8 − [B6 × (1 − s) + B11 × s],  s = (B8 − B6)/(B11 − B6)",
        "What it indicates": "Alternative floating-algae index with a red-edge/SWIR baseline (model input)",
        "Map layer": "No",
        "Reference": None,
    },
    {
        "Index": "MAI",
        "Formula (as implemented)": "(B8 − B5) / (B8 + B5)",
        "What it indicates": "Micro-algae index",
        "Map layer": "Yes",
        "Reference": None,
    },
    {
        "Index": "TBMAI",
        "Formula (as implemented)": "(B6 − B5) × (B8 − B5) / (B6 + B5)",
        "What it indicates": "Red-edge / NIR band-combination index (model input)",
        "Map layer": "No",
        "Reference": None,
    },
    {
        "Index": "Chl-a (estimated)",
        "Formula (as implemented)": "35.75 × (B8 / B4) − 14.85   [µg/L]",
        "What it indicates": "Empirical chlorophyll-a estimate",
        "Map layer": "Yes",
        "Reference": None,
    },
    {
        "Index": "TSI",
        "Formula (as implemented)": "Predicted per pixel by the selected model from the 17 inputs",
        "What it indicates": "Carlson Trophic State Index (0-100) and trophic class",
        "Map layer": "Yes (heatmap and points)",
        "Reference": "Carlson (1977)",
    },
]

REFERENCES = [
    "Carlson, R. E. (1977). A trophic state index for lakes. Limnology and Oceanography, 22(2), 361-369.",
    "Hu, C. (2009). A novel ocean color index to detect floating algae in the global oceans. "
    "Remote Sensing of Environment, 113(10), 2118-2129.",
    "Mishra, S., & Mishra, D. R. (2012). Normalized difference chlorophyll index: A novel model for "
    "remote estimation of chlorophyll-a concentration in turbid productive waters. "
    "Remote Sensing of Environment, 117, 394-406.",
    "Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water "
    "features in remotely sensed imagery. International Journal of Remote Sensing, 27(14), 3025-3033.",
]