import json
import logging
import shutil
from pathlib import Path

from src.local_assets import ensure_local_assets


logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)


if __name__ == "__main__":
    ensure_local_assets()

    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    file_name = "run_information.json"

    with open(root_path / file_name, "r", encoding="utf-8") as f:
        run_info = json.load(f)
        logger.info("Information loaded successfully")

    registry_dir = root_path / "models" / "registry" / "staging"
    registry_dir.mkdir(parents=True, exist_ok=True)

    source_model_path = Path(run_info["model_uri"])
    registered_model_path = registry_dir / "uber_demand_prediction_model.joblib"
    shutil.copy2(source_model_path, registered_model_path)

    run_info["registered_model_path"] = str(registered_model_path)
    run_info["registered_stage"] = "Staging"
    with open(root_path / file_name, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4)

    logger.info("Model registered locally in staging")
