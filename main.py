import argparse
import logging
from sentence_transformers import SentenceTransformer
import torch
from src.utils.helpers import setup_logging, load_config
from src.validation.validate_lyrics import load_artist_data

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def load_lyrics_from_file(file_path):
    """Load lyrics from a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading lyrics file: {e}")
        raise


def analyze_lyrics(lyrics_path):
    """Analyze lyrics and compare with artist embeddings."""
    # Load config and data
    config = load_config()
    artist_embeddings, artist_names = load_artist_data()

    # Load the lyrics
    lyrics = load_lyrics_from_file(lyrics_path)

    # Initialize the model
    model = SentenceTransformer(config["model"]["model_name"])

    # Get embedding for the lyrics
    lyrics_embedding = model.encode(lyrics, convert_to_numpy=True)

    # Calculate similarities with all artist embeddings
    similarity_scores = model.similarity(lyrics_embedding, artist_embeddings)[0]

    # Get top 5 most similar artists
    top_indices = torch.argsort(similarity_scores, descending=True)[:3]
    top_artists = [artist_names[idx] for idx in top_indices]
    top_scores = similarity_scores[top_indices]

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("LYRICS SIMILARITY ANALYSIS")
    logger.info("=" * 50)

    for artist, score in zip(top_artists, top_scores):
        logger.info(f"{artist}: {score:.4%}")

    logger.info("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze lyrics similarity to artists")
    parser.add_argument("lyrics_file", type=str, help="Path to the lyrics text file")
    args = parser.parse_args()

    try:
        analyze_lyrics(args.lyrics_file)
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
