"""Lake Trophic Status (TSI) — public web app.

Run locally with:   streamlit run app.py
Deploy with:         Streamlit Community Cloud, or any host that runs
                      `streamlit run app.py` (Cloud Run, a small VM, etc.)

Setup required before first run — see README.md:
  1. A GEE service account key, stored in .streamlit/secrets.toml
  2. Trained model file(s) placed in the models/ folder
"""
import datetime as dt
import glob
import os
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from lake_tsi.gee_utils import build_median_composite, download_composite_direct, initialise_gee_service_account
from lake_tsi.mapping import build_interactive_map
from lake_tsi.models import load_model, predict_tsi
from lake_tsi.raster_utils import load_tiff_to_dataframe, save_tsi_heatmap_tiff
from lake_tsi.shapefile_utils import load_lake_geometry_from_zip
from lake_tsi.trophic import overall_trophic

st.set_page_config(page_title="Lake Trophic Status", layout="wide")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def available_models() -> dict:
    """Model files bundled by the developer in models/, shown as a dropdown."""
    paths = glob.glob(os.path.join(MODELS_DIR, "*.pkl")) + \
        glob.glob(os.path.join(MODELS_DIR, "*.joblib")) + \
        glob.glob(os.path.join(MODELS_DIR, "*.pth")) + \
        glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    return {os.path.basename(p): p for p in paths}


st.title("🌊 Lake Trophic Status — Sentinel-2 + Earth Engine")
st.caption(
    "Upload a lake boundary, choose a date range and a trained model, "
    "and get an interactive map of eutrophication status."
)

with st.sidebar:
    st.header("Inputs")

    lake_name = st.text_input("Lake name", value="")
    col_a, col_b = st.columns(2)
    with col_a:
        study_year = st.text_input("Study year", value=str(dt.date.today().year))
    with col_b:
        season = st.selectbox("Season", ["Pre-Monsoon", "Post-Monsoon", "Winter", "Summer", "Other"])

    st.divider()
    shapefile_zip = st.file_uploader(
        "Lake boundary shapefile (.zip containing .shp/.shx/.dbf/.prj)", type=["zip"]
    )

    st.divider()
    date_range = st.date_input(
        "Sentinel-2 date range",
        value=(dt.date.today() - dt.timedelta(days=90), dt.date.today()),
    )
    cloud_pct = st.slider("Max cloud cover (%)", 0, 100, 20)
    scale = st.select_slider("Export resolution (m/pixel)", options=[10, 20, 30, 60], value=20)

    st.divider()
    models = available_models()
    if not models:
        st.warning("No model files found in models/. Add a .pkl/.joblib/.pth file there.")
    model_choice = st.selectbox("Trophic-state model", list(models.keys()) if models else ["(none available)"])

    st.divider()
    run_button = st.button("Run analysis", type="primary", use_container_width=True)


if run_button:
    if not shapefile_zip:
        st.error("Please upload a shapefile (.zip) first.")
        st.stop()
    if not models:
        st.error("No model available — add one to the models/ folder and reload.")
        st.stop()
    if len(date_range) != 2:
        st.error("Please select a start and end date.")
        st.stop()

    start_date, end_date = str(date_range[0]), str(date_range[1])
    work_dir = tempfile.mkdtemp()
    tiff_path = os.path.join(work_dir, "composite.tif")
    tsi_tiff_path = os.path.join(work_dir, "tsi_heatmap.tif")
    map_html_path = os.path.join(work_dir, "map.html")

    try:
        with st.status("Running analysis…", expanded=True) as status:
            status.write("Authenticating with Earth Engine…")
            initialise_gee_service_account()

            status.write("Reading shapefile…")
            geometry, gdf = load_lake_geometry_from_zip(shapefile_zip)

            status.write(f"Fetching Sentinel-2 imagery ({start_date} to {end_date})…")
            median_img, n_images = build_median_composite(geometry, start_date, end_date, cloud_pct)
            status.write(f"Found {n_images} cloud-filtered image(s). Downloading composite…")

            download_composite_direct(median_img, geometry, tiff_path, scale=scale)

            status.write("Clipping to lake boundary and extracting pixels…")
            pixel_df = load_tiff_to_dataframe(tiff_path, gdf)
            if len(pixel_df) == 0:
                raise RuntimeError(
                    "No valid pixels found inside the shapefile boundary — check that "
                    "the shapefile and imagery overlap."
                )
            status.write(f"{len(pixel_df):,} valid pixels extracted.")

            status.write(f"Loading model ({model_choice}) and predicting TSI…")
            model, kind = load_model(models[model_choice])
            result_df, uncertainty = predict_tsi(pixel_df, model, kind)
            overall = overall_trophic(result_df)

            status.write("Building TSI heatmap raster…")
            save_tsi_heatmap_tiff(result_df, gdf, tsi_tiff_path, resolution=512)

            status.write("Rendering interactive map…")
            build_interactive_map(
                gdf=gdf, result_df=result_df, overall=overall,
                tsi_tiff_path=tsi_tiff_path, out_html=map_html_path,
                lake_name=lake_name or "Lake", study_year=study_year, season=season,
            )

            status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state["map_html_path"] = map_html_path
        st.session_state["overall"] = overall
        st.session_state["result_df"] = result_df
        st.session_state["uncertainty_mean"] = float(uncertainty.mean()) if uncertainty is not None else None

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()


if "map_html_path" in st.session_state and os.path.exists(st.session_state["map_html_path"]):
    overall = st.session_state["overall"]
    result_df = st.session_state["result_df"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TSI Score", f"{overall['TSI_Score']:.1f}")
    m2.metric("Trophic Status", overall["Trophic_Status"])
    m3.metric("Pixels analysed", f"{len(result_df):,}")
    unc = st.session_state.get("uncertainty_mean")
    m4.metric("Mean model uncertainty", f"±{unc:.2f}" if unc is not None else "n/a")

    with open(st.session_state["map_html_path"], "r", encoding="utf-8") as f:
        map_html = f.read()
    components.html(map_html, height=800, scrolling=False)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download pixel-level results (CSV)", csv_bytes, "tsi_results.csv", "text/csv")
else:
    st.info("Fill in the inputs on the left and click **Run analysis** to generate the map.")
