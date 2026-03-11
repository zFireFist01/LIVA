from  palmtree.config import *
from palmtree.eval_utils import UsableTransformer
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

import enum
import os

# PalmTree import modules
import numpy as np


# Word embedding model parameters
EMBEDDING_SIZE = 128


class DistanceMetric(enum.Enum):
    """Distance metrics for embeddings"""

    BRAYCURTIS: str = "braycurtis"
    CHEBYSHEV: str = "chebyshev"
    CORRELATION: str = "correlation"
    COSINE: str = "cosine"
    EUCLIDEAN: str = "euclidean"
    SQEUCLIDEAN: str = "sqeuclidean"


class ModelNotInitializedError(Exception):
    """Exception raised when trying to use a `Model` object that is not initialized"""


class PalmTree:
    """Word embedding model"""

    def __init__(self, name: str = ""):
        self._model = None  # Word embedding model
        self.name = name.strip()  # Word embedding model name

    def __repr__(self) -> str:
        return "<{}.{}; name='{}'>".format(__name__, type(self).__name__, self.name)

    def __str__(self) -> str:
        return "'{}': {!r}".format(self.name, self._model)

    def _check(self):
        """Raise `ModelNotInitializedError` if the word embedding model is not initialized"""

        if not self._model:
            raise ModelNotInitializedError("The word embedding model is not initialized")

    def load(self, file_path: str):
        """Load a trained word embedding model from disk"""

        # Check file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError("Can't find '{}'".format(file_path))
        
        vocab_path = Path(__file__).parent / "palmtree" / "model" / "vocab"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Can't find '{vocab_path}'")

        print(vocab_path)
        self._model = UsableTransformer(model_path=file_path, vocab_path=vocab_path)

    def save(self, file_path: str):
        """Save the trained word embedding model to disk"""

        self._check()

        self._model.save_model(file_path)

    def get_embedding(self, instructions: list[str]) -> np.ndarray[np.float32]:
        """Get an embedding from a sentence"""
        # TODO to be further modified to resemble Gemini
        
        self._check()
        
        if len(instructions) == 0:
            return None

        window = 500    # parameter for the context of split of instruction
        context = 20   # how much context to give to the model

        i_embed = self._model.encode(instructions[:window+context])[:window]
        for i in range(window, len(instructions), window):
            i_embed = np.concatenate((i_embed, self._model.encode(instructions[i:i+window+context])[:window]), axis=0)

        return i_embed


def get_embeddings_similarity(source_embedding: np.ndarray, target_embedding: np.ndarray, distance_metric: DistanceMetric) -> float:
    """Get the similarity score between two embeddings using a distance metric"""

    if source_embedding is None or target_embedding is None:
        similarity = -1
    elif np.array_equal(source_embedding, target_embedding):
        similarity = 1
    else:
        similarity = cosine_similarity(source_embedding, target_embedding).item(0)

    return float(similarity)


def compute_similarity_matrix(source_functions, target_functions):
    source_embeddings = np.stack([np.squeeze(f.embedding) for f in source_functions])  # shape (m, d)
    target_embeddings = np.stack([np.squeeze(f.embedding) for f in target_functions])  # shape (n, d)

    # Use sklearn for optimized cosine similarity
    similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)  # shape (m, n)

    return similarity_matrix
