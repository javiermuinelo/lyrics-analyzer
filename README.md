# Lyrics Style Analysis & Generation

## Description

This project analyzes song lyrics using natural language processing techniques to understand artists' unique writing styles and generate new lyrics that match their style. It uses sentence transformers to create embeddings of artists' lyrics and can validate generated lyrics against learned artist styles.

## Features

- Preprocesses and cleans raw lyrics data
- Creates artist-specific embeddings using sentence transformers
- Analyzes similarity between input lyrics and artist styles
- Generates new lyrics in the style of specific artists using OpenAI's API
- Validates generated lyrics against learned artist styles

## Raw GENIUS Dataset

The raw GENIUS dataset is available at: https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information
If you want to run the preprocessing steps and create the embeddings, you can download the dataset csv and put the csv file in the `data/raw` folder.

## Project Structure
```
.
├── configs/
│   └── config.yaml          # Configuration settings
├── data/
│   ├── raw/                 # Raw lyrics GENIUS dataset
│   ├── interim/             # Intermediate processed data
│   ├── processed/           # Final processed dataset
│   └── generated/           # AI-generated validation lyrics
├── src/
│   ├── data/               # Data processing scripts
│   ├── features/           # Embedding creation
│   ├── validation/         # Validation scripts
│   ├── utils/              # Utility functions
│   └── notebooks/          # Jupyter notebooks for analysis
├── main.py                 # Main script for lyrics analysis
└── requirements.txt        # Project dependencies
```

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/lyrics-style-analysis.git
cd lyrics-style-analysis
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key (only if you want to generate validation lyrics):

```bash
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Main Usage: Analyzing Lyrics Similarity

To analyze how similar input lyrics are to different artists' styles:

```bash
python main.py path/to/your/lyrics.txt
```

This will output similarity scores between your input lyrics and each artist's style in the database.

Example output:
```
Similarity Scores:
Taylor Swift: 0.82
Ed Sheeran: 0.75
The Beatles: 0.65
...
```

### Additional Tools

1. Data Preprocessing:

```bash
python src/data/preprocess.py
```

2. Generate Artist Embeddings:

```bash
python src/features/build_features.py
```

3. Generate Validation Lyrics:

```bash
python src/data/generate_validation_lyrics.py
```

4. Validate Generated Lyrics:

```bash
python src/validation/validate_lyrics.py
```

## Configuration

The project configuration is managed through `configs/config.yaml`. Key settings include:

- Data file paths
- Model parameters
- List of anchor artists
- Preprocessing options

## Notebooks

The project includes Jupyter notebooks for exploratory data analysis and debugging:

- `notebooks/data_exploration.ipynb`: Initial data analysis and preprocessing steps
- `notebooks/artist_embeddings.ipynb`: Analysis of artist embeddings and style similarity
