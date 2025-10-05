# promote_mlflow.py
"""
Promote the latest (or exact) MLflow model version to the 'Staging' stage.
Used in GitHub Actions CI/CD pipeline.
"""

import os
from mlflow.tracking import MlflowClient


# --- Load configuration from environment variables ---
URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.ml.brain.cs.ait.ac.th/")
NAME  = os.getenv("MLFLOW_MODEL_NAME", None)  # ✅ FIXED: no longer KeyError
MODE  = os.getenv("PROMOTE_MODE", "latest")   # "latest" or "exact"
EXACT = os.getenv("MLFLOW_MODEL_VERSION")     # used if MODE="exact"

USER  = os.getenv("MLFLOW_TRACKING_USERNAME")
PWD   = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not NAME:
    raise ValueError("❌ MLFLOW_MODEL_NAME environment variable not set. Please define it in your workflow.")

if USER:
    os.environ["MLFLOW_TRACKING_USERNAME"] = USER
if PWD:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = PWD


# --- Connect to MLflow ---
print(f"🔗 Connecting to MLflow tracking server at: {URI}")
client = MlflowClient(tracking_uri=URI)


# --- Helper to get latest version ---
def pick_latest_version(name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"❌ No versions found for model '{name}' — check MLflow model registry.")
    latest = max(versions, key=lambda v: int(v.version)).version
    return latest


# --- Determine which version to promote ---
version = EXACT if MODE == "exact" else pick_latest_version(NAME)

print(f"🚀 Promoting model '{NAME}' version {version} → Staging ...")
client.transition_model_version_stage(
    name=NAME,
    version=version,
    stage="Staging",
    archive_existing_versions=False
)
print("✅ Promotion complete.")