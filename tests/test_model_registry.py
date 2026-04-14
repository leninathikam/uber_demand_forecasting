import joblib

from src.local_assets import ensure_local_assets


def test_load_model_from_registry():
    model_path = ensure_local_assets()["model"]
    model = joblib.load(model_path)
    assert model is not None, "Failed to load model from local registry"
