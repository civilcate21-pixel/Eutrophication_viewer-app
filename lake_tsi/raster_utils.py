"""TIFF <-> DataFrame conversion and TSI heatmap raster generation.

Ported from notebook cells 6 and 9. Logic unchanged.
"""
import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from shapely.geometry import mapping

from .gee_utils import FEATURE_NAMES


def load_tiff_to_dataframe(tiff_path: str, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Clip a multi-band TIFF to the lake boundary and return every valid pixel
    as a DataFrame with latitude, longitude + band columns. No sampling."""
    with rasterio.open(tiff_path) as src:
        n_bands = src.count
        gdf_proj = gdf.to_crs(src.crs)
        shapes = [mapping(geom) for geom in gdf_proj.geometry]
        arr, transform = rio_mask(src, shapes, crop=True, nodata=np.nan)
        arr = arr.astype(np.float32)

        height, width = arr.shape[1], arr.shape[2]
        rows_idx, cols_idx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = rasterio.transform.xy(transform, rows_idx.ravel(), cols_idx.ravel())
        lons = np.array(xs)
        lats = np.array(ys)

    data = arr.reshape(n_bands, -1).T
    cols = FEATURE_NAMES if n_bands == len(FEATURE_NAMES) else [f"band_{i+1}" for i in range(n_bands)]

    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "longitude", lons)
    df.insert(0, "latitude", lats)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_tsi_heatmap_tiff(
    result_df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    out_path: str,
    resolution: int = 512,
    method: str = "linear",
) -> str:
    lons = result_df["longitude"].values
    lats = result_df["latitude"].values
    tsi = result_df["TSI_Score"].values

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

    grid_tsi = griddata(
        points=np.column_stack([lons, lats]), values=tsi,
        xi=np.column_stack([gx.ravel(), gy.ravel()]), method=method,
    ).reshape(resolution, resolution).astype(np.float32)

    nan_mask = np.isnan(grid_tsi)
    if nan_mask.any():
        grid_tsi_nn = griddata(
            points=np.column_stack([lons, lats]), values=tsi,
            xi=np.column_stack([gx.ravel(), gy.ravel()]), method="nearest",
        ).reshape(resolution, resolution).astype(np.float32)
        grid_tsi[nan_mask] = grid_tsi_nn[nan_mask]

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, resolution, resolution)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    with rasterio.open(
        tmp_path, "w", driver="GTiff", height=resolution, width=resolution,
        count=1, dtype=np.float32, crs=CRS.from_epsg(4326), transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(grid_tsi, 1)

    gdf_wgs = gdf.to_crs(epsg=4326)
    shapes = [mapping(geom) for geom in gdf_wgs.geometry]

    with rasterio.open(tmp_path) as src:
        clipped, clip_transform = rio_mask(src, shapes, crop=True, nodata=np.nan, filled=True)
        clip_meta = src.meta.copy()

    clip_meta.update({
        "height": clipped.shape[1], "width": clipped.shape[2],
        "transform": clip_transform, "dtype": np.float32,
        "nodata": np.nan, "compress": "lzw",
    })

    with rasterio.open(out_path, "w", **clip_meta) as dst:
        dst.write(clipped[0].astype(np.float32), 1)
        dst.update_tags(description="TSI Heatmap - Carlson Trophic State Index", units="TSI (0-100+)")

    os.remove(tmp_path)
    return out_path
