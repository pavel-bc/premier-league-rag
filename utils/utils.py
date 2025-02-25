import random
import requests
import numpy as np

# Model used for embeddings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDINGS_MODEL = "mxbai-embed-large"


# Random match prediction fallback
def random_match_prediction():
    return random.randint(0, 5), random.randint(0, 5)


# Match embedding text
def match_embedding_text(home_team, away_team, home_goals, away_goals, season_year):
    return f"{home_team} vs {away_team} {home_goals}:{away_goals} ({season_year})"


def get_ollama_embedding(text):
    """Get embeddings from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDINGS_MODEL, "prompt": text},
        )
        return np.array(response.json()["embedding"])
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return np.zeros(1024)  # Return zero vector as fallback
