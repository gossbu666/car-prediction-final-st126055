# promote_mlflow.py
"""
Promote a registered MLflow model to "staging".
- Prefers MLflow 3.x style ALIASES (staging/production).
- Falls back to older STAGES API if available.
"""

import os
import urllib3
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Silence TLS warnings if we're told to skip verification
if os.getenv("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in ("1", "true", "yes"):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URI   = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th/")
NAME  = os.getenv("MLFLOW_MODEL_NAME")                 # e.g., st126055-a3-model
MODE  = os.getenv("PROMOTE_MODE", "latest")            # "latest" or "exact"
EXACT = os.getenv("MLFLOW_MODEL_VERSION")              # used if MODE="exact"
ALIAS = os.getenv("MLFLOW_ALIAS_NAME", "staging")      # default alias name

USER  = os.getenv("MLFLOW_TRACKING_USERNAME")
PWD   = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not NAME:
    raise ValueError("MLFLOW_MODEL_NAME is not set.")
if USER is None or PWD is None:
    raise ValueError("Missing MLflow credentials. Set repo secrets MLFLOW_USER and MLFLOW_PASS.")

# Pass creds to MLflow client
os.environ["MLFLOW_TRACKING_USERNAME"] = USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = PWD

print(f"🔗 Connecting to MLflow server: {URI}")
client = MlflowClient(tracking_uri=URI)

def pick_latest_version(name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"❌ No versions found for model '{name}'. Is it registered with any versions?")
    # choose max numeric version
    return max(versions, key=lambda v: int(v.version)).version

version = EXACT if MODE == "exact" else pick_latest_version(NAME)
print(f"📦 Target model: {NAME}  |  version: {version}")

# ---- Try old 'stages' first (if server supports it), then fall back to 'aliases'
try:
    print(f"🧪 Trying legacy stage transition → Staging …")
    # NOTE: deprecated in MLflow ≥2.9, may be disabled on the server
    client.transition_model_version_stage(
        name=NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print("✅ Promotion via STAGES complete.")
except Exception as e:
    print(f"ℹ️ Stage transition failed ({type(e).__name__}): {e}")
    print(f"➡️ Falling back to ALIASES: setting alias '{ALIAS}' → version {version}")
    try:
        # Set/overwrite alias (atomic re-point)
        client.set_registered_model_alias(name=NAME, alias=ALIAS, version=str(version))
        # Optional: show where the alias points now
        resolved = client.get_model_version_by_alias(name=NAME, alias=ALIAS)
        print(f"✅ Alias set: {NAME}@{ALIAS} → v{resolved.version}")
    except MlflowException as e2:
        raise SystemExit(f"❌ Alias promotion failed: {e2}")