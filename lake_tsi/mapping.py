"""Interactive folium map builder - ported from the notebook's build_interactive_map()
and its helper functions. Logic and HTML/JS untouched; only imports reorganised."""
import base64
import io
import json
import os

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium import Element
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds as fb
from scipy.interpolate import griddata
from shapely.geometry import mapping

from .trophic import TROPHIC_CLASSES, TROPHIC_MARKER


def add_ee_layer(map_obj, ee_img, vis_params, name, show=True):
    try:
        import ee
        mid = ee.Image(ee_img).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=mid["tile_fetcher"].url_format,
            attr="Google Earth Engine", name=name,
            overlay=True, control=True, show=show,
        ).add_to(map_obj)
    except Exception as e:
        print(f"[!] Could not add EE layer '{name}': {e}")


def build_index_overlay(map_obj, result_df: pd.DataFrame, gdf: gpd.GeoDataFrame,
                         column: str, layer_name: str, colormap_colors: list,
                         vmin: float = None, vmax: float = None,
                         resolution: int = 512, opacity: float = 0.7) -> folium.Map:
    lons = result_df["longitude"].values
    lats = result_df["latitude"].values
    values = result_df[column].values

    v_min = vmin if vmin is not None else float(np.nanpercentile(values, 2))
    v_max = vmax if vmax is not None else float(np.nanpercentile(values, 98))

    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()
    pad_lon = (max_lon - min_lon) * 0.01 or 0.001
    pad_lat = (max_lat - min_lat) * 0.01 or 0.001
    min_lon -= pad_lon
    max_lon += pad_lon
    min_lat -= pad_lat
    max_lat += pad_lat

    grid_lon = np.linspace(min_lon, max_lon, resolution)
    grid_lat = np.linspace(max_lat, min_lat, resolution)
    gx, gy = np.meshgrid(grid_lon, grid_lat)

    grid = griddata(
        points=np.column_stack([lons, lats]), values=values,
        xi=np.column_stack([gx.ravel(), gy.ravel()]), method="linear",
    ).reshape(resolution, resolution)

    nan_mask = np.isnan(grid)
    if nan_mask.any():
        grid_nn = griddata(
            points=np.column_stack([lons, lats]), values=values,
            xi=np.column_stack([gx.ravel(), gy.ravel()]), method="nearest",
        ).reshape(resolution, resolution)
        grid[nan_mask] = grid_nn[nan_mask]

    gdf_wgs = gdf.to_crs(epsg=4326)
    mask_transform = fb(min_lon, min_lat, max_lon, max_lat, resolution, resolution)
    lake_mask = geometry_mask(
        [mapping(geom) for geom in gdf_wgs.geometry],
        transform=mask_transform, invert=True, out_shape=(resolution, resolution),
    )

    cmap = LinearSegmentedColormap.from_list(column + "_cmap", colormap_colors)
    grid_norm = np.clip((grid - v_min) / (v_max - v_min + 1e-10), 0.0, 1.0)
    rgba = (cmap(grid_norm) * 255).astype(np.uint8)
    rgba[~lake_mask, 3] = 0

    buf = io.BytesIO()
    PILImage.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{b64}",
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=opacity, name=layer_name, overlay=True, control=True, show=False,
    ).add_to(map_obj)
    return map_obj


def build_tsi_heatmap_overlay(map_obj, tsi_tiff_path: str, gdf: gpd.GeoDataFrame, opacity: float = 0.75):
    import rasterio

    tsi_colors = [
        (0/100, "#2196F3"), (30/100, "#2196F3"), (30/100, "#4CAF50"), (40/100, "#4CAF50"),
        (40/100, "#8BC34A"), (50/100, "#8BC34A"), (50/100, "#FFC107"), (60/100, "#FFC107"),
        (60/100, "#FF5722"), (70/100, "#FF5722"), (70/100, "#B71C1C"), (1.0, "#B71C1C"),
    ]
    cmap = LinearSegmentedColormap.from_list("tsi_correct", tsi_colors)

    with rasterio.open(tsi_tiff_path) as src:
        arr = src.read(1).astype(float)
        bounds = src.bounds
        nodata = src.nodata

    gdf_wgs = gdf.to_crs(epsg=4326)
    h, w = arr.shape
    mask_transform = fb(bounds.left, bounds.bottom, bounds.right, bounds.top, w, h)
    lake_mask = geometry_mask(
        [mapping(geom) for geom in gdf_wgs.geometry],
        transform=mask_transform, invert=True, out_shape=(h, w),
    )

    arr_norm = np.clip(arr, 0.0, 100.0) / 100.0
    rgba = (cmap(arr_norm) * 255).astype(np.uint8)

    outside = ~lake_mask
    if nodata is not None:
        outside = outside | (arr == nodata) | np.isnan(arr)
    else:
        outside = outside | np.isnan(arr)
    rgba[outside, 3] = 0

    pil_img = PILImage.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{b64}",
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=opacity, name="TSI Heatmap", overlay=True, control=True, show=True,
    ).add_to(map_obj)
    return map_obj


def build_interactive_map(gdf: gpd.GeoDataFrame, result_df: pd.DataFrame, overall: dict,
                           tsi_tiff_path: str, median_img=None, geometry=None,
                           out_html: str = "lake_trophic_map.html", lake_name: str = "Lake",
                           study_year: str = "", season: str = "") -> folium.Map:

    bounds = gdf.total_bounds
    centre = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    Map = folium.Map(location=centre, zoom_start=13, tiles="CartoDB positron")

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google", name="Google Satellite", overlay=False, control=True,
    ).add_to(Map)

    index_layers = [
        ("MNDWI", "MNDWI (Water Index)",
         ["#ffffff", "#b3e5fc", "#4fc3f7", "#0288d1", "#01579b", "#002244"], -0.3, 0.8),
        ("NDCI", "NDCI (Chlorophyll Index)",
         ["#1a0066", "#1400ff", "#00bbff", "#00ff44", "#ffee00", "#ff0000"], -0.1, 0.5),
        ("CHLA", "Estimated Chl-a (ug/L)",
         ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff8800", "#ff0000"], 0, 80),
        ("FAI", "FAI (Floating Algae Index)",
         ["#000080", "#0000ff", "#00ffff", "#7fff00", "#ffff00", "#ff0000"], -0.05, 0.1),
        ("MAI", "MAI (Micro Algae Index)",
         ["#000080", "#0000ff", "#00ffff", "#7fff00", "#ffff00", "#ff0000"], -0.2, 0.3),
        ("CI_Rededge", "CI Red-Edge (Phytoplankton)",
         ["#000000", "#004400", "#00ff00", "#ffff00", "#ff8800", "#ff0000"], 0, 3),
    ]

    for col, name, colors, vmin, vmax in index_layers:
        if col in result_df.columns:
            build_index_overlay(Map, result_df, gdf, col, name, colors, vmin, vmax, resolution=512)

    if tsi_tiff_path and os.path.exists(tsi_tiff_path):
        build_tsi_heatmap_overlay(Map, tsi_tiff_path, gdf, opacity=0.75)

    for cid, tc in enumerate(TROPHIC_MARKER):
        sub = result_df[result_df["Class_ID"] == cid]
        if len(sub) == 0:
            continue
        fg = folium.FeatureGroup(name=f"TSI Points: {tc['label']}", show=True, control=False)
        for _, row in sub.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3, color=tc["color"], fill=True, fill_color=tc["color"], fill_opacity=0.85,
                popup=(f"<b>TSI:</b> {row['TSI_Score']:.1f}<br>"
                       f"<b>Status:</b> {row['Trophic_Status']}<br>"
                       f"<b>Chl-a:</b> {row.get('CHLA', 0):.1f} ug/L"),
                tooltip=f"TSI {row['TSI_Score']:.1f}",
            ).add_to(fg)
        fg.add_to(Map)

    folium.GeoJson(
        gdf.to_json(), name="Lake Boundary",
        style_function=lambda x: {"color": "#00E5FF", "weight": 3, "fillColor": "transparent", "opacity": 0.9},
    ).add_to(Map)

    def _fmt(v, dp=4):
        return f"{v:.{dp}f}"

    stats = {
        "NDCI": {"min": result_df["NDCI"].min(), "max": result_df["NDCI"].max(), "mean": result_df["NDCI"].mean(),
                 "palette": PALETTE_NDCI, "unit": "", "label": "NDCI", "layer": "NDCI (Chlorophyll Index)"},
        "CHLA": {"min": result_df["CHLA"].min(), "max": result_df["CHLA"].max(), "mean": result_df["CHLA"].mean(),
                 "palette": PALETTE_CHLA, "unit": " ug/L", "label": "Chl-a", "layer": "Estimated Chl-a (ug/L)"},
        "MAI": {"min": result_df["MAI"].min(), "max": result_df["MAI"].max(), "mean": result_df["MAI"].mean(),
                "palette": PALETTE_FAI, "unit": "", "label": "MAI", "layer": "MAI (Micro Algae Index)"},
        "FAI": {"min": result_df["FAI"].min(), "max": result_df["FAI"].max(), "mean": result_df["FAI"].mean(),
                "palette": PALETTE_FAI, "unit": "", "label": "FAI", "layer": "FAI (Floating Algae Index)"},
        "MNDWI": {"min": result_df["MNDWI"].min(), "max": result_df["MNDWI"].max(), "mean": result_df["MNDWI"].mean(),
                  "palette": PALETTE_MNDWI, "unit": "", "label": "MNDWI", "layer": "MNDWI (Water Index)"},
        "CI_Rededge": {"min": result_df["CI_Rededge"].min(), "max": result_df["CI_Rededge"].max(),
                       "mean": result_df["CI_Rededge"].mean(), "palette": PALETTE_CIREDEGE, "unit": "",
                       "label": "CI Red-Edge", "layer": "CI Red-Edge (Phytoplankton)"},
    }

    legend_rows_html = ""
    for tc in TROPHIC_CLASSES:
        is_active = (tc["label"] == overall["Trophic_Status"] or
                     overall["Trophic_Status"].startswith(tc["label"].split("-")[0]))
        border = f"2px solid {tc['color']}" if is_active else "2px solid transparent"
        bg = "rgba(255,255,255,0.08)" if is_active else "transparent"
        legend_rows_html += f"""
        <div style="display:flex;align-items:center;margin:3px 0;
                    padding:4px 6px;border-radius:6px;
                    border:{border};background:{bg};">
          <div style="width:16px;height:16px;border-radius:3px;flex-shrink:0;
                      background:{tc['color']};margin-right:9px;
                      box-shadow:0 0 5px {tc['color']}88;"></div>
          <div>
            <div style="font-size:11px;font-weight:600;color:#fff;line-height:1.2;">
              {tc['label']}</div>
            <div style="font-size:9px;color:#90CAF9;line-height:1.2;">
              {tc['range']} &nbsp;.&nbsp; {tc['short']}</div>
          </div>
        </div>"""

    trophic_legend_html = f"""
    <div id="trophic-legend" style="
        position:fixed;bottom:30px;right:20px;z-index:9999;
        background:rgba(8,16,36,0.93);border-radius:12px;padding:14px 16px;
        font-family:'Segoe UI',Arial,sans-serif;
        box-shadow:0 4px 28px rgba(0,0,0,0.6);
        min-width:205px;max-width:305px;
        border:1px solid rgba(144,202,249,0.2);">
      <div style="display:flex;align-items:center;margin-bottom:10px;">
        <div style="width:4px;height:28px;border-radius:2px;
                    background:linear-gradient(#2196F3,#B71C1C);
                    margin-right:10px;flex-shrink:0;"></div>
        <div>
          <div style="font-size:10px;letter-spacing:2px;color:#90CAF9;
                      text-transform:uppercase;font-weight:600;">
            Trophic State Legend</div>
          <div style="font-size:9px;color:#607D8B;margin-top:1px;">
            Carlson TSI . Sentinel-2 derived</div>
        </div>
      </div>
      {legend_rows_html}
      <div style="margin-top:10px;">
        <div style="font-size:9px;color:#607D8B;margin-bottom:3px;
                    text-transform:uppercase;letter-spacing:1px;">TSI Scale</div>
        <div style="height:10px;border-radius:5px;
                    background:linear-gradient(to right,
                      #2196F3 0%,#4CAF50 30%,#8BC34A 40%,
                      #FFC107 50%,#FF5722 60%,#B71C1C 70%);
                    border:1px solid rgba(255,255,255,0.15);"></div>
        <div style="display:flex;justify-content:space-between;
                    font-size:8px;color:#90CAF9;margin-top:2px;">
          <span>0</span><span>30</span><span>40</span>
          <span>50</span><span>60</span><span>70</span><span>100</span>
        </div>
      </div>
    </div>"""

    meta_chips = ""
    if study_year:
        meta_chips += f"""
        <span style="background:rgba(33,150,243,0.15);border:1px solid rgba(33,150,243,0.3);
                    border-radius:20px;padding:2px 10px;font-size:10px;color:#90CAF9;">
         {study_year}</span>"""
    if season:
        meta_chips += f"""
        <span style="background:rgba(129,199,132,0.15);border:1px solid rgba(129,199,132,0.3);
                    border-radius:20px;padding:2px 10px;font-size:10px;color:#A5D6A7;">
         {season}</span>"""

    title_html = f"""
    <div id="map-title" style="
        position:fixed;top:16px;left:50%;transform:translateX(-50%);z-index:9999;
        background:rgba(8,16,36,0.93);border-radius:12px;padding:10px 22px;
        font-family:'Segoe UI',Arial,sans-serif;
        box-shadow:0 4px 28px rgba(0,0,0,0.55);
        border:1px solid rgba(144,202,249,0.2);
        text-align:center;white-space:nowrap;">
    <div style="font-size:15px;font-weight:700;color:#fff;letter-spacing:0.4px;
                margin-bottom:5px;">
         {lake_name}
        <span style="font-size:11px;font-weight:400;color:#90CAF9;margin-left:6px;">
        - Eutrophication Status
        </span>
    </div>
    <div style="display:flex;gap:8px;justify-content:center;align-items:center;">
        <span style="background:rgba(144,202,249,0.1);border:1px solid rgba(144,202,249,0.25);
                    border-radius:20px;padding:2px 10px;font-size:10px;color:#80CBC4;">
        Carlson TSI</span>
        {meta_chips}
    </div>
    </div>"""

    def _gradient_css(palette):
        n = len(palette)
        stops = ", ".join(f"{c} {round(i / (n - 1) * 100)}%" for i, c in enumerate(palette))
        return f"linear-gradient(to right, {stops})"

    container_html = """
    <div id="legend-container" style="
        position: fixed;
        top: 50px;
        left: 20px;
        z-index: 9998;
        display: flex;
        flex-direction: column;
        gap: 10px;
    ">
    </div>
    """

    index_legends_html = ""
    layer_legend_map = {}

    for key, s in stats.items():
        grad = _gradient_css(s["palette"])
        vmin = _fmt(s["min"])
        vmean = _fmt(s["mean"])
        vmax = _fmt(s["max"])
        unit = s["unit"]
        layer_name = s["layer"]
        legend_id = f"idx_legend_{key}"
        layer_legend_map[layer_name] = legend_id

        index_legends_html += f"""
        <div id="{legend_id}" style="
            display: none;
            background:rgba(8,16,36,0.93);
            border-radius:6px;
            padding:6px 10px;
            font-family:'Segoe UI',Arial,sans-serif;
            box-shadow:0 4px 28px rgba(0,0,0,0.6);
            min-width:220px;
            border:1px solid rgba(144,202,249,0.2);">

            <div style="font-size:10px;color:#90CAF9;margin-bottom:8px;">
                {s["label"]} Legend
            </div>

            <div style="height:14px;border-radius:5px;margin-bottom:4px;
                        background:{grad};"></div>

            <div style="display:flex;justify-content:space-between;font-size:9px;color:#90CAF9;">
                <span>Min: {vmin}{unit}</span>
                <span>Max: {vmax}{unit}</span>
            </div>

            <div style="font-size:10px;color:#fff;margin-top:4px;">
                Mean: {vmean}{unit}
            </div>
        </div>
        """

    js_watcher = f"""
    <script>
    (function() {{
        const layerLegendMap = {json.dumps(layer_legend_map)};

        function normalize(text) {{
            return text.replace(/\\s+/g, " ").trim();
        }}

        function syncIndexLegends() {{
            const checkedLayers = new Set();
            const overlayContainer = document.querySelector('.leaflet-control-layers-overlays');

            if (overlayContainer) {{
                const checkboxes = overlayContainer.querySelectorAll('input[type="checkbox"]');
                checkboxes.forEach(checkbox => {{
                    if (checkbox.checked) {{
                        const label = checkbox.closest('label');
                        if (label) {{
                            const labelText = normalize(label.textContent);
                            checkedLayers.add(labelText);
                        }}
                    }}
                }});
            }}

            const container = document.getElementById('legend-container');
            if (!container) return;

            container.innerHTML = '';

            let hasAnyLegend = false;
            for (const [layerName, legendId] of Object.entries(layerLegendMap)) {{
                if (checkedLayers.has(normalize(layerName))) {{
                    const legendDiv = document.getElementById(legendId);
                    if (legendDiv) {{
                        const clonedLegend = legendDiv.cloneNode(true);
                        clonedLegend.style.display = 'block';
                        container.appendChild(clonedLegend);
                        hasAnyLegend = true;
                    }}
                }}
            }}

            container.style.display = hasAnyLegend ? 'flex' : 'none';
        }}

        function setupLayerControlWatcher() {{
            const checkInterval = setInterval(function() {{
                const overlayContainer = document.querySelector('.leaflet-control-layers-overlays');
                if (overlayContainer) {{
                    clearInterval(checkInterval);

                    const checkboxes = overlayContainer.querySelectorAll('input[type="checkbox"]');
                    checkboxes.forEach(checkbox => {{
                        checkbox.addEventListener('change', syncIndexLegends);
                    }});

                    const observer = new MutationObserver(function(mutations) {{
                        mutations.forEach(function(mutation) {{
                            if (mutation.addedNodes.length) {{
                                const newCheckboxes = overlayContainer.querySelectorAll('input[type="checkbox"]');
                                newCheckboxes.forEach(checkbox => {{
                                    if (!checkbox.hasListener) {{
                                        checkbox.addEventListener('change', syncIndexLegends);
                                        checkbox.hasListener = true;
                                    }}
                                }});
                            }}
                        }});
                    }});

                    observer.observe(overlayContainer, {{ childList: true, subtree: true }});
                    syncIndexLegends();
                }}
            }}, 500);
        }}

        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', setupLayerControlWatcher);
        }} else {{
            setupLayerControlWatcher();
        }}
    }})();
    </script>
    """

    tsi = round(overall["TSI_Score"], 2)
    status = overall["Trophic_Status"]
    detail = overall["Detail"]
    rec = overall["Recommendation"]
    col = overall.get("Color", "#2196F3")
    if col == "#2196F3" and tsi >= 30:
        for tc in TROPHIC_CLASSES:
            if tc["label"] == status:
                col = tc["color"]
                break
    mean_chla = result_df["CHLA"].mean()
    mean_fai = result_df["FAI"].mean()
    mean_mai = result_df["MAI"].mean()
    mean_ndci = result_df["NDCI"].mean()
    tsi_min = result_df["TSI_Score"].min()
    tsi_max = result_df["TSI_Score"].max()
    bar_pct = min(100, max(0, tsi))

    panel_html = f"""
    <div id="tsi-panel" style="
        position:fixed;bottom:30px;left:20px;z-index:9999;
        background:rgba(8,16,36,0.93);border-radius:12px;padding:16px 20px;
        font-family:'Segoe UI',Arial,sans-serif;
        box-shadow:0 4px 28px rgba(0,0,0,0.6);
        min-width:260px;max-width:310px;
        border:1px solid rgba(144,202,249,0.2);
        border-left:4px solid {col};">
      <div style="font-size:10px;letter-spacing:2px;color:#90CAF9;
                  text-transform:uppercase;font-weight:600;margin-bottom:8px;">
        Lake Trophic Analysis</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px;">
        <div style="font-size:36px;font-weight:800;color:{col};
                    line-height:1;text-shadow:0 0 20px {col}66;">{tsi:.2f}</div>
        <div style="font-size:11px;color:#90CAF9;">TSI Score</div>
      </div>
      <div style="height:6px;border-radius:3px;margin-bottom:10px;
                  background:rgba(255,255,255,0.1);overflow:hidden;">
        <div style="height:100%;width:{bar_pct}%;border-radius:3px;
                    background:linear-gradient(to right,#2196F3,{col});"></div>
      </div>
      <div style="font-size:13px;font-weight:700;color:#fff;margin-bottom:2px;">
        {status}</div>
      <div style="font-size:11px;color:#b0bec5;margin-bottom:10px;">{detail}</div>
      <hr style="border:none;border-top:1px solid rgba(55,71,79,0.8);margin:8px 0;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;
                  font-size:11px;margin-bottom:10px;">
        <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;">
          <div style="color:#80DEEA;font-size:9px;margin-bottom:2px;">Chl-a (mean)</div>
          <div style="color:#fff;font-weight:600;">{mean_chla:.1f} ug/L</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;">
          <div style="color:#80DEEA;font-size:9px;margin-bottom:2px;">FAI (mean)</div>
          <div style="color:#fff;font-weight:600;">{mean_fai:.4f}</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;">
          <div style="color:#80DEEA;font-size:9px;margin-bottom:2px;">MAI (mean)</div>
          <div style="color:#fff;font-weight:600;">{mean_mai:.4f}</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;">
          <div style="color:#80DEEA;font-size:9px;margin-bottom:2px;">NDCI (mean)</div>
          <div style="color:#fff;font-weight:600;">{mean_ndci:.4f}</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;">
          <div style="color:#80DEEA;font-size:9px;margin-bottom:2px;">TSI min</div>
          <div style="color:#fff;font-weight:600;">{tsi_min:.1f}</div>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;">
          <div style="color:#80DEEA;font-size:9px;margin-bottom:2px;">TSI max</div>
          <div style="color:#fff;font-weight:600;">{tsi_max:.1f}</div>
        </div>
      </div>
      <div style="background:rgba(255,204,128,0.08);border-radius:6px;
                  padding:8px 10px;border-left:3px solid #FFCC80;">
        <div style="font-size:9px;color:#FFCC80;font-weight:700;
                    text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">
          Recommendation</div>
        <div style="font-size:10px;color:#FFE0B2;line-height:1.4;">{rec}</div>
      </div>
    </div>"""

    Map.get_root().html.add_child(Element(title_html))
    Map.get_root().html.add_child(Element(trophic_legend_html))
    Map.get_root().html.add_child(Element(container_html))
    Map.get_root().html.add_child(Element(index_legends_html))
    Map.get_root().html.add_child(Element(js_watcher))
    Map.get_root().html.add_child(Element(panel_html))
    folium.LayerControl().add_to(Map)
    Map.save(out_html)
    return Map


# Palettes re-exported here for stats dict construction above
from .trophic import PALETTE_CHLA, PALETTE_NDCI, PALETTE_MNDWI, PALETTE_FAI, PALETTE_CIREDEGE  # noqa: E402
