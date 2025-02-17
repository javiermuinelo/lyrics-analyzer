import yaml
from pathlib import Path
import logging
from typing import Dict
from dotenv import load_dotenv


def get_project_root() -> Path:
    """Get the absolute path to project root directory."""
    current_file = Path(__file__)
    return current_file.parent.parent.parent


def load_env_variables() -> None:
    """Load environment variables from .env file."""
    project_root = get_project_root()
    env_path = project_root / ".env"

    if not env_path.exists():
        raise FileNotFoundError(
            f".env file not found at {env_path}. "
            "Please create one with your OPENAI_API_KEY."
        )

    load_dotenv(env_path)


def load_config() -> Dict:
    """Load configuration from yaml file."""
    project_root = get_project_root()
    config_path = project_root / "configs" / "config.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise


def setup_logging(log_path: Path = None) -> None:
    """Setup logging configuration.

    Args:
        log_path (Path, optional): Path where logs will be saved.
                                  If None, only console logging is setup.
    """
    handlers = [logging.StreamHandler()]

    if log_path:
        log_path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path / "app.log"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# Initialize logger after defining setup function
logger = logging.getLogger(__name__)
