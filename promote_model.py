import json
import shutil
from pathlib import Path

from src.local_assets import ROOT_PATH, ensure_local_assets


def load_model_information(file_path: Path) -> dict:
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    ensure_local_assets()
    run_info_path = ROOT_PATH / "run_information.json"
    run_info = load_model_information(run_info_path)

    staging_model_path = Path(
        run_info.get("registered_model_path", ROOT_PATH / "models" / "model.joblib")
    )
    production_dir = ROOT_PATH / "models" / "registry" / "production"
    production_dir.mkdir(parents=True, exist_ok=True)
    production_model_path = production_dir / "uber_demand_prediction_model.joblib"
    shutil.copy2(staging_model_path, production_model_path)

    run_info["production_model_path"] = str(production_model_path)
    run_info["registered_stage"] = "Production"
    run_info_path.write_text(json.dumps(run_info, indent=4), encoding="utf-8")

    print("The model is moved to the local Production stage")
