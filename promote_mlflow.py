# promote_mlflow.py
"""
Promote the latest (or exact) MLflow model version to the 'Staging' stage.
Used by GitHub Actions. Handles HTTPS + optional self-signed TLS.
"""

import os
from mlflow.tracking import MlflowClient

# --- Config from environment ---
URI   = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th/")
NAME  = os.getenv("MLFLOW_MODEL_NAME")
MODE  = os.getenv("PROMOTE_MODE", "latest")   # "latest" or "exact"
EXACT = os.getenv("MLFLOW_MODEL_VERSION")     # used if MODE="exact"

USER  = os.getenv("MLFLOW_TRACKING_USERNAME")
PWD   = os.getenv("MLFLOW_TRACKING_PASSWORD")

# allow skipping TLS verification (self-signed certs)
if os.getenv("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in ("1", "true", "yes"):
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

if not NAME:
    raise ValueError("MLFLOW_MODEL_NAME is not set.")
if USER is None or PWD is None:
    raise ValueError("Missing MLflow credentials. Set repo secrets MLFLOW_USER and MLFLOW_PASS.")

# forward creds for mlflow client to pick up
os.environ["MLFLOW_TRACKING_USERNAME"] = USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = PWD

print(f"🔗 Connecting to MLflow tracking server at: {URI}")
client = MlflowClient(tracking_uri=URI)

def pick_latest_version(name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{name}'. Is it registered with versions?")
    latest = max(versions, key=lambda v: int(v.version)).version
    return latest

version = EXACT if MODE == "exact" else pick_latest_version(NAME)

print(f"🚀 Promoting model '{NAME}' version {version} → Staging ...")
# note: this API is deprecated in mlflow 2.9+, but still works; acceptable for class assignment
client.transition_model_version_stage(
    name=NAME,
    version=version,
    stage="Staging",
    archive_existing_versions=False
)
print("✅ Promotion complete.")