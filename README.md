# Lake Trophic Status — Web App

Streamlit web app for the GEE + Sentinel-2 lake trophic-state (TSI) pipeline.
Users upload a lake boundary shapefile, pick a date range and a trained
model, and get back an interactive map — all inside the browser.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Add your trained model(s)

Copy your trained model files (`.pkl`, `.joblib`, `.pth`, or `.pt`) into the
`models/` folder. Each one shows up as an option in the app's model
dropdown. (Retrain offline as before — the app only loads finished models,
it never trains.)

⚠️ Your BNN training script currently does **not** save the
`StandardScaler` used to normalize inputs before training. If your BNN
model was trained on scaled features, save the scaler too
(`joblib.dump(scaler, "models/scaler.pkl")`) and scale `pixel_df` the same
way before prediction — otherwise BNN predictions will be wrong. This app's
`load_model`/`predict_tsi` currently assumes the model was trained on raw
(unscaled) feature values, matching your RandomForest/XGBoost training.

## 3. Set up a GEE service account (required for public use)

1. In Google Cloud Console, create a service account under your GEE-enabled
   project and grant it the **Earth Engine Resource Viewer** role (or
   broader, if it needs to read private assets).
2. Register that service account for Earth Engine access at
   https://signup.earthengine.google.com/#!/service_accounts
3. Create and download a JSON key for the service account.
4. Create `.streamlit/secrets.toml` in the project root:

```toml
[gee]
service_account_email = "my-app@my-project.iam.gserviceaccount.com"
project_id = "my-project"
private_key_json = '''
{ ...paste the full contents of the downloaded JSON key file here... }
'''
```

Never commit `secrets.toml` to git — add it to `.gitignore`.

## 4. Run locally

```bash
streamlit run app.py
```

## 5. Deploy

**Streamlit Community Cloud (simplest, free tier):**
1. Push this repo to GitHub (excluding `secrets.toml`).
2. Go to https://share.streamlit.io, connect the repo, set `app.py` as the
   entry point.
3. In the app's Settings → Secrets, paste the same `[gee]` block from step 3.

**Anything else that runs `streamlit run app.py`** (Cloud Run, a VM,
Docker) works too — just make sure the same secrets are available as
environment/secret config at runtime.

## Notes / limits to plan for

- **GEE quotas**: public traffic can hit Earth Engine compute/download
  limits faster than a single-user notebook would. Watch for rate-limit
  errors under load.
- **Large lakes**: `download_composite_direct()` downloads the composite
  directly via `getDownloadURL` (no Google Drive round-trip). This works
  well for typical lake extents at 20 m/px, but very large lakes or very
  fine resolutions may exceed GEE's per-request download size limit and
  need tiling — flag this if you hit it.
- **Shapefile upload**: must be a `.zip` containing `.shp` + `.shx` +
  `.dbf` (and ideally `.prj`) — a lone `.shp` file isn't valid on its own.
- **Model swapped mid-session**: switching the model dropdown and re-running
  re-predicts on the same downloaded imagery pixel data, so re-runs after
  the first are fast.
