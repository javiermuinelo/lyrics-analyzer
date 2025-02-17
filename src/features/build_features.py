import pandas as pd
import pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import sys
import os
import logging

sys.path.append(".")
from src.utils.helpers import load_config
import logging

logger = logging.getLogger(__name__)


class ArtistEmbeddingCreator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding creator with a Sentence Transformer model.

        Args:
            model_name (str): Name of the sentence-transformers model to use
                            Some good options:
                            - 'all-MiniLM-L6-v2' (fast & good quality)
                            - 'all-mpnet-base-v2' (better quality but slower)
                            - 'all-distilroberta-v1' (good balance)
        """
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def create_song_embedding(self, lyrics: str) -> np.ndarray:
        """Create embedding for a single song by encoding the full lyrics as one string."""
        # Encode the full lyrics as a single string
        embedding = self.model.encode(
            lyrics,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return embedding

    def create_artist_embedding(self, artist_lyrics: List[str]) -> np.ndarray:
        """Create a single embedding vector for an artist from all their songs."""
        song_embeddings = []

        for lyrics in tqdm(artist_lyrics, desc="Processing songs"):
            embedding = self.create_song_embedding(lyrics)
            song_embeddings.append(embedding)

        # Average all song embeddings to get artist embedding
        artist_embedding = np.mean(song_embeddings, axis=0)
        return artist_embedding


def create_artist_embeddings(
    df: pd.DataFrame,
    save_path: Path = None,
    model_name: str = None,
    validation_path: Path = None,
) -> Dict[str, np.ndarray]:
    """Create embeddings for all artists in the dataset."""
    creator = ArtistEmbeddingCreator(model_name=model_name)
    artist_embeddings = {}
    artists = []

    for artist in tqdm(df["artist"].unique(), desc="Processing artists"):
        artist_lyrics = df[df["artist"] == artist]["lyrics"].tolist()
        if validation_path is not None:
            validation_artist_lyrics = artist_lyrics[:5]
            # save validation artist lyrics as rows of 'artist' and 'lyrics' in a csv file
            validation_df = pd.DataFrame(
                {"artist": artist, "lyrics": validation_artist_lyrics}
            )

            # If file exists, append without headers, otherwise create new file with headers
            if os.path.exists(validation_path):
                validation_df.to_csv(
                    validation_path, mode="a", header=False, index=False
                )
            else:
                validation_df.to_csv(
                    validation_path, index=False, columns=["artist", "lyrics"]
                )

            artist_lyrics = artist_lyrics[5:]

        artist_embeddings[artist] = creator.create_artist_embedding(artist_lyrics)
        artists.append(artist)

    if save_path:
        # Convert embeddings to a matrix where each row is an artist embedding
        embedding_matrix = np.array([artist_embeddings[artist] for artist in artists])

        # Save the embedding matrix as .npy file
        logger.info("Saving artist embeddings...")
        np.save(save_path + "_embeddings.npy", embedding_matrix)

        # Save artist names as pickle file
        logger.info("Saving artist names...")
        with open(save_path + "_names.pkl", "wb") as f:
            pickle.dump(artists, f)

    return embedding_matrix, artists


if __name__ == "__main__":
    config = load_config()

    logger.info("Loading processed data...")
    df = pd.read_csv(config["data"]["processed_file"])

    model_name = config["model"]["model_name"]

    logger.info("Filtering anchor artists...")
    anchor_artists = config["anchor_artists"]
    df = df[df["artist"].isin(anchor_artists)]

    logger.info("Creating artist embeddings...")
    create_artist_embeddings(
        df,
        config["embeddings"]["path"],
        model_name,
        validation_path=config["validation"]["path"],
    )
