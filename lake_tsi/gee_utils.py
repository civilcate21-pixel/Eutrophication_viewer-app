"""Google Earth Engine session handling and Sentinel-2 feature extraction.

Key change from the source notebook: `initialise_gee_service_account()`
replaces `ee.Authenticate()` (interactive browser login) with a service
account key, so it works headless on a server for anonymous public users.
"""
import json
import os

import ee
import streamlit as st

FEATURE_NAMES = [
    "Blue", "Green", "Red", "RE1", "RE2", "RE3",
    "NIR", "SWIR",
    "MNDWI", "NDCI", "CI_Rededge", "Hyph",
    "FAI", "AFAI", "MAI", "TBMAI", "CHLA",
]


@st.cache_resource(show_spinner=False)
def initialise_gee_service_account() -> None:
    """Authenticate to GEE using a service account key stored in st.secrets.

    Required in .streamlit/secrets.toml (or the Streamlit Cloud secrets UI):

        [gee]
        service_account_email = "my-app@my-project.iam.gserviceaccount.com"
        project_id = "my-project"
        private_key_json = '''
        { ... full contents of the downloaded JSON key file ... }
        '''

    Cached with st.cache_resource so the credential handshake only happens
    once per server process, not once per user request.
    """
    gee_cfg = st.secrets["gee"]
    key_dict = json.loads(gee_cfg["private_key_json"])

    credentials = ee.ServiceAccountCredentials(
        gee_cfg["service_account_email"], key_data=json.dumps(key_dict)
    )
    ee.Initialize(credentials, project=gee_cfg["project_id"])


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    m = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return (
        image.updateMask(m)
        .divide(10000)
        .divide(3.14159265)
        .copyProperties(image, ["system:time_start"])
    )


def get_sentinel2_collection(geometry: ee.Geometry, start: str, end: str, cloud_pct: int) -> ee.ImageCollection:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_s2_clouds)
        .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11"])
    )
    count = col.size().getInfo()
    if count == 0:
        raise RuntimeError(
            "No Sentinel-2 images found for this date range / cloud threshold. "
            "Try widening the date range or raising the cloud %."
        )
    return col, count


def add_spectral_indices(image: ee.Image) -> ee.Image:
    blue, green, red = image.select("B2"), image.select("B3"), image.select("B4")
    re1, re2, re3 = image.select("B5"), image.select("B6"), image.select("B7")
    nir, swir = image.select("B8"), image.select("B11")
    eps = 1e-10

    mndwi = red.subtract(swir).divide(red.add(swir).add(eps)).rename("MNDWI")
    ndci = re2.subtract(re1).divide(re2.add(re1).add(eps)).rename("NDCI")
    ci_re = nir.subtract(re1).divide(nir.add(re1).add(eps)).rename("CI_Rededge")
    hyph = red.subtract(re1).divide(red.add(re1).add(eps)).rename("Hyph")
    nir_p = red.add(swir.subtract(red).multiply((842 - 665) / (1610 - 665)))
    fai = nir.subtract(nir_p).rename("FAI")
    sl = nir.subtract(re2).divide(swir.subtract(re2).add(eps))
    afai = nir.subtract(
        re2.multiply(ee.Image.constant(1).subtract(sl)).add(swir.multiply(sl))
    ).rename("AFAI")
    mai = nir.subtract(re1).divide(nir.add(re1).add(eps)).rename("MAI")
    tbmai = re2.subtract(re1).multiply(nir.subtract(re1)).divide(re2.add(re1).add(eps)).rename("TBMAI")
    chla = nir.divide(red.add(eps)).multiply(35.75).subtract(14.85).rename("CHLA")

    return (
        image.addBands(mndwi).addBands(ndci).addBands(ci_re).addBands(hyph)
        .addBands(fai).addBands(afai).addBands(mai).addBands(tbmai).addBands(chla)
        .rename(FEATURE_NAMES)
    )


def build_median_composite(geometry: ee.Geometry, start: str, end: str, cloud_pct: int):
    """Fetch S2 collection, add indices, return (median_image, image_count)."""
    col, count = get_sentinel2_collection(geometry, start, end, cloud_pct)
    all_imgs = col.map(add_spectral_indices)
    median_img = all_imgs.mean()
    return median_img, count


def download_composite_direct(median_img: ee.Image, geometry: ee.Geometry, out_path: str, scale: int = 20) -> str:
    """Download the clipped composite straight to a local GeoTIFF.

    Replaces the notebook's Export.image.toDrive + manual-download flow,
    which requires a human at a Drive UI.

    NOTE: this used to call geemap.ee_export_image(), which wraps the
    download in its own try/except and *prints* failures to stdout instead
    of raising - meaning real errors (request-too-large, auth issues, etc.)
    were invisible in the Streamlit UI and only a generic message showed up.
    This version calls ee.Image.getDownloadURL() directly and raises the
    real HTTP/GEE error message so failures are actually visible to the user.
    """
    import requests

    image = median_img.clip(geometry).toFloat()

    try:
        url = image.getDownloadURL({
            "scale": scale,
            "crs": "EPSG:4326",
            "region": geometry,
            "format": "GEO_TIFF",
        })
    except Exception as e:
        raise RuntimeError(f"Earth Engine rejected the download request: {e}")

    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code != 200:
        # GEE puts the real reason (e.g. "Total request size must be less
        # than 50331648 bytes") in the response body - surface it directly.
        raise RuntimeError(
            f"GEE download failed (HTTP {resp.status_code}): {resp.text[:500]}"
        )

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("Download completed but the file is empty — no data was returned for this region/date range.")

    return out_path
