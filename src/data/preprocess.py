import pandas as pd
import re
import logging
from utils.helpers import get_project_root, load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_lyrics(lyrics):
    """Clean lyrics text by removing section tags, symbols, and extra whitespace."""
    # Remove section tags like [Intro], [Verse 1], etc.
    cleaned = re.sub(r"\[.*?\]", "", lyrics)

    # Remove credits or text after "---"
    cleaned = re.split(r"---", cleaned)[0]

    # Remove symbols except for line breaks (\n) and alphanumeric characters
    cleaned = re.sub(r"[^\w\s\n]", "", cleaned)

    # Remove extra whitespace and blank lines
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned).strip()

    return cleaned


def preprocess_data():
    """Main function to preprocess the lyrics data."""
    try:
        config = load_config()
        project_root = get_project_root()

        # Create Path objects
        raw_data_path = config["data"]["raw_file"]
        interim_data_path = config["data"]["interim_file"]
        processed_data_path = config["data"]["processed_file"]

        # Ensure directories exist
        interim_data_path.parent.mkdir(parents=True, exist_ok=True)
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)

        # Load raw data
        logger.info("Loading raw data...")
        raw_data = pd.read_csv(raw_data_path)
        logger.info(f"Loaded {len(raw_data)} rows of raw data")

        # Filter English lyrics
        logger.info("Filtering English lyrics...")
        english_lyrics = raw_data[raw_data["language"] == "en"]
        english_lyrics.to_csv(interim_data_path, index=False)
        logger.info(f"Saved {len(english_lyrics)} English lyrics to interim data")

        # Remove songs with features
        if config["preprocessing"]["remove_features"]:
            logger.info("Removing songs with features...")
            english_lyrics = english_lyrics[english_lyrics["features"] == "{}"]

        # Drop unnecessary columns
        logger.info("Dropping unnecessary columns...")
        columns_to_drop = [
            "views",
            "features",
            "language",
            "language_cld3",
            "language_ft",
            "id",
            "year",
        ]
        english_lyrics = english_lyrics.drop(columns=columns_to_drop)

        # Clean lyrics
        if config["preprocessing"]["clean_lyrics"]:
            logger.info("Cleaning lyrics text...")
            english_lyrics["lyrics"] = english_lyrics["lyrics"].apply(clean_lyrics)

        # Save processed data
        logger.info("Saving processed data...")
        english_lyrics.to_csv(processed_data_path, index=False)
        logger.info(
            f"Saved {len(english_lyrics)} processed lyrics to {processed_data_path}"
        )

        return True

    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise


if __name__ == "__main__":
    preprocess_data()
