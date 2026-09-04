"""Load a lake boundary from a user-uploaded zipped shapefile.

A bare .shp is not usable on its own (needs .shx/.dbf/.prj sidecars), so the
web app requires a single .zip containing the full shapefile set. Ported
from the notebook's load_lake_geometry(), swapping the local-path argument
for an uploaded file object.
"""
import tempfile
import zipfile
from pathlib import Path

import ee
import geopandas as gpd


def load_lake_geometry_from_zip(uploaded_zip) -> tuple[ee.Geometry, gpd.GeoDataFrame]:
    """uploaded_zip: a Streamlit UploadedFile (from st.file_uploader)."""
    tmp_dir = tempfile.mkdtemp()
    zip_path = Path(tmp_dir) / "shapefile.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    shp_files = list(Path(tmp_dir).rglob("*.shp"))
    if not shp_files:
        raise ValueError(
            "No .shp file found inside the uploaded zip. Make sure the zip "
            "contains the .shp, .shx, .dbf (and ideally .prj) files together."
        )

    gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
    merged_poly = gdf.geometry.union_all()
    geometry = ee.Geometry(merged_poly.__geo_interface__)
    return geometry, gdf
