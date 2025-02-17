import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging
import sys
import pickle

sys.path.append(".")
from src.utils.helpers import get_project_root, setup_logging, load_config

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def load_validation_data(config):
    """Load the generated validation lyrics from CSV."""
    project_root = get_project_root()
    validation_path = project_root / config["validation"]["path"]

    try:
        df = pd.read_csv(validation_path)
        return df
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        raise


def load_artist_data():
    """Load the artist embeddings and names."""
    project_root = get_project_root()
    config = load_config()

    # Get embeddings directory from config
    embeddings_path = project_root / (config["embeddings"]["path"] + "_embeddings.npy")
    names_path = project_root / (config["embeddings"]["path"] + "_names.pkl")

    try:
        # Load artist embeddings from .npy file
        artist_embeddings = np.load(embeddings_path)
        # Load artist names from pickle
        with open(names_path, "rb") as f:
            artist_names = pickle.load(f)

        return artist_embeddings, artist_names
    except Exception as e:
        logger.error(f"Error loading artist data: {e}")
        raise


def validate_lyrics():
    """Validate generated lyrics against artist embeddings."""
    # Load config
    config = load_config()

    # Load the data
    validation_df = load_validation_data(config)
    artist_embeddings, artist_names = load_artist_data()

    # Initialize the sentence transformer model using config
    model_name = config["model"]["model_name"]
    model = SentenceTransformer(model_name)

    results = []
    artist_results = {}
    top1_correct = 0
    top3_correct = 0
    total = 0

    for _, row in validation_df.iterrows():
        true_artist = row["artist"]
        lyrics = row["lyrics"]

        # Initialize artist results if not exists
        if true_artist not in artist_results:
            artist_results[true_artist] = {
                "total": 0,
                "top1_correct": 0,
                "top3_correct": 0,
            }

        # Get embedding for the lyrics
        lyrics_embedding = model.encode(lyrics, convert_to_numpy=True)

        # Calculate similarities with all artist embeddings
        similarity_scores = model.similarity(lyrics_embedding, artist_embeddings)[0]

        # Get top 3 most similar artists
        top_indices = torch.argsort(similarity_scores, descending=True)[:3]
        top_artists = [artist_names[idx] for idx in top_indices]
        top_scores = similarity_scores[top_indices]

        # Check if true artist is in top 1 and top 3
        is_top1 = top_artists[0] == true_artist
        is_top3 = true_artist in top_artists

        # Update global counters
        top1_correct += int(is_top1)
        top3_correct += int(is_top3)
        total += 1

        # Update artist-specific counters
        artist_results[true_artist]["total"] += 1
        artist_results[true_artist]["top1_correct"] += int(is_top1)
        artist_results[true_artist]["top3_correct"] += int(is_top3)

        result = {
            "true_artist": true_artist,
            "predicted_artists": top_artists,
            "similarity_scores": top_scores.tolist(),
            "correct_top1": is_top1,
            "correct_top3": is_top3,
        }
        results.append(result)

    # Print formatted results
    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION RESULTS PER ARTIST")
    logger.info("=" * 50)

    # Sort artists by their top-1 accuracy for better visualization
    sorted_artists = sorted(
        artist_results.items(),
        key=lambda x: x[1]["top1_correct"] / x[1]["total"] if x[1]["total"] > 0 else 0,
        reverse=True,
    )

    for artist, stats in sorted_artists:
        total_artist = stats["total"]
        if total_artist > 0:  # Avoid division by zero
            top1_acc = stats["top1_correct"] / total_artist
            top3_acc = stats["top3_correct"] / total_artist
            logger.info(f"\n{artist} (Total samples: {total_artist})")
            logger.info(f"├── Top-1 Accuracy: {top1_acc:.2%}")
            logger.info(f"└── Top-3 Accuracy: {top3_acc:.2%}")

    # Calculate and print global accuracies
    top1_accuracy = top1_correct / total
    top3_accuracy = top3_correct / total

    logger.info("\n" + "=" * 50)
    logger.info("GLOBAL VALIDATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total samples: {total}")
    logger.info(f"├── Global Top-1 Accuracy: {top1_accuracy:.2%}")
    logger.info(f"└── Global Top-3 Accuracy: {top3_accuracy:.2%}")
    logger.info("=" * 50 + "\n")

    return results


if __name__ == "__main__":
    validate_lyrics()
