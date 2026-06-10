from palmtree.eval_utils import UsableTransformer
from pathlib import Path

import os

import numpy as np

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
