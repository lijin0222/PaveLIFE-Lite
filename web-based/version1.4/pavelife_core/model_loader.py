# pavelife_core/model_loader.py

import os

from pavelife_core import legacy


def build_rate_model_artifact_paths(model_dir):
    """Build artifact paths for asphalt and concrete pretrained rate models."""
    artifact_paths = {
        "asphalt": {
            "model_path": os.path.join(model_dir, "model_xgb_asphalt.joblib"),
            "feature_columns_path": os.path.join(model_dir, "feature_columns_asphalt.joblib"),
            "encoder_path": os.path.join(model_dir, "onehot_encoder_asphalt.joblib"),
            "scaler_path": os.path.join(model_dir, "scaler_asphalt.joblib"),
        },
        "concrete": {
            "model_path": os.path.join(model_dir, "model_xgb_concrete.joblib"),
            "feature_columns_path": os.path.join(model_dir, "feature_columns_concrete.joblib"),
            "encoder_path": os.path.join(model_dir, "onehot_encoder_concrete.joblib"),
            "scaler_path": os.path.join(model_dir, "scaler_concrete.joblib"),
        },
    }
    return artifact_paths


def set_model_artifact_dir(model_dir):
    """Update the model-artifact directory used by the legacy calculation layer."""
    model_dir = os.path.abspath(model_dir)
    legacy.pretrained_model_dir = model_dir
    legacy.rate_model_artifacts = build_rate_model_artifact_paths(model_dir)
    return legacy.rate_model_artifacts


def load_pretrained_rate_artifacts(pavement_family, model_dir=None):
    """Load one pavement-family-specific pretrained rate model."""
    pavement_family = str(pavement_family).lower().strip()

    if model_dir is not None:
        set_model_artifact_dir(model_dir)

    return legacy.load_pretrained_rate_artifacts(pavement_family)


def load_all_model_artifacts(model_dir=None):
    """Load asphalt and concrete model artifacts."""
    if model_dir is not None:
        set_model_artifact_dir(model_dir)

    model_artifacts = {}
    for pavement_family in ["asphalt", "concrete"]:
        model_artifacts[pavement_family] = legacy.load_pretrained_rate_artifacts(pavement_family)

    return model_artifacts
