import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = "configs/app_config.yaml") -> dict:
    cfg_path = PROJECT_ROOT / config_path
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")
    return key
