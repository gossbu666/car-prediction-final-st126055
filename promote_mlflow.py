import os
from mlflow.tracking import MlflowClient

URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.ml.brain.cs.ait.ac.th/")
NAME  = os.environ["st126055-a3-model"]            # e.g., st126055-a3-model
MODE  = os.getenv("PROMOTE_MODE", "latest")        # "latest" or "exact"
EXACT = os.getenv("MLFLOW_MODEL_VERSION")          # used if MODE="exact"

user = os.getenv("MLFLOW_TRACKING_USERNAME")
pwd  = os.getenv("MLFLOW_TRACKING_PASSWORD")
if user: os.environ["MLFLOW_TRACKING_USERNAME"] = user
if pwd:  os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd

client = MlflowClient(tracking_uri=URI)

def pick_latest_version(name: str) -> str:
    vers = client.search_model_versions(f"name='{name}'")
    if not vers:
        raise RuntimeError(f"No versions found for model {name}")
    return max(vers, key=lambda v: int(v.version)).version

version = EXACT if MODE == "exact" else pick_latest_version(NAME)
print(f"Promoting model {NAME} v{version} → Staging")
client.transition_model_version_stage(
    name=NAME, version=version, stage="Staging", archive_existing_versions=False
)
print("✅ Promotion complete.")