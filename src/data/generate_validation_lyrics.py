import os
import openai
from typing import List, Dict
import logging
import pandas as pd
import os

import sys

sys.path.append(".")
from src.utils.helpers import (
    load_config,
    setup_logging,
    get_project_root,
    load_env_variables,
)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def get_openai_client():
    """Setup OpenAI API with environment variables."""
    # Load environment variables from .env file
    load_env_variables()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file. " "Please add it to your .env file."
        )
    client = openai.OpenAI(api_key=api_key)
    return client


def generate_lyrics(
    client: openai.OpenAI,
    artist: str,
    example_lyrics: pd.DataFrame,
    num_songs: int = 10,
) -> List[Dict]:
    """
    Generate lyrics in the style of a specific artist using OpenAI API.

    Args:
        artist (str): Name of the artist
        num_songs (int): Number of songs to generate

    Returns:
        List[Dict]: List of generated songs with metadata
    """
    generated_songs = []

    for i in range(num_songs):
        try:
            # Sample 2 random lyrics
            sample_lyrics = example_lyrics["lyrics"].sample(n=2).tolist()

            prompt = (
                f"Write a new, original song lyric in the style of {artist}. Use similar structure, rhyme scheme and tone as the artist's songs. "
                f"Here are some examples of {artist}'s songs:\n"
                f"Example 1: {sample_lyrics[0][:150]}...\n"
                f"Example 2: {sample_lyrics[1][:150]}...\n"
                "The song should be unique and not copy any existing songs. "
                "Write only the pure lyrics without any structural indicators. "
                "Start the lyrics with 'LYRICS:' and end them with 'END_LYRICS' "
                "on separate lines."
            )

            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "You are a creative songwriter."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=300,
            )

            generated_text = response.choices[0].message.content.strip()

            # Extract lyrics between markers
            try:
                lyrics = (
                    generated_text.split("LYRICS:")[1].split("END_LYRICS")[0].strip()
                )
            except IndexError:
                logger.warning(
                    f"Generated text for {artist} did not contain expected markers, skipping"
                )
                continue

            song_data = {
                "artist": artist,
                "lyrics": lyrics,
            }

            generated_songs.append(song_data)
            logger.info(f"Generated song {i+1}/{num_songs} for {artist}")

        except Exception as e:
            logger.error(f"Error generating song {i+1} for {artist}: {e}")
            continue

    return generated_songs


def save_generated_lyrics(songs: List[Dict], artist: str):
    """Save generated lyrics to a CSV file."""
    project_root = get_project_root()
    output_dir = project_root / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / "validation_lyrics.csv"

    try:
        # Create DataFrame from songs list
        df = pd.DataFrame(songs)

        # Append to CSV if it exists, create new one if it doesn't
        if filename.exists():
            df.to_csv(filename, mode="a", header=False, index=False)
        else:
            df.to_csv(
                filename,
                index=False,
                columns=["artist", "lyrics"],
            )

        logger.info(f"Saved generated lyrics for {artist} to {filename}")
    except Exception as e:
        logger.error(f"Error saving lyrics for {artist}: {e}")


def main():
    """Main function to generate lyrics for all artists."""
    try:
        # Load configuration
        config = load_config()
        artists = config.get("anchor_artists", [])
        project_root = get_project_root()

        artists_example_lyrics = pd.read_csv(
            project_root / config["data"]["processed_file"]
        )

        if not artists:
            logger.error("No artists found in config file")
            return

        # Setup OpenAI API
        client = get_openai_client()

        # Generate lyrics for each artist
        for artist in artists:
            example_lyrics = artists_example_lyrics[
                artists_example_lyrics["artist"] == artist
            ]
            logger.info(f"Generating lyrics for {artist}")
            generated_songs = generate_lyrics(
                client, artist, example_lyrics, num_songs=5
            )
            save_generated_lyrics(generated_songs, artist)
            logger.info(f"Completed generating lyrics for {artist}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
