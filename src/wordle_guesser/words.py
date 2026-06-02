"""Word-list loading and a compact vocabulary representation.

v1 uses the official answer list (~2315 words) as both the candidate set and
the model's output vocabulary. The allowed-guess list is loaded too for future
use (full-probe solver) but is not required by the v1 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
WORD_LEN = 5


def load_word_list(path: str | Path) -> list[str]:
    """Read a newline-separated word list, keeping only 5-letter [a-z] words."""
    words: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            w = line.strip().lower()
            if len(w) == WORD_LEN and w.isalpha() and w.isascii():
                words.append(w)
    if not words:
        raise ValueError(f"no valid 5-letter words found in {path}")
    return words


def words_to_letters(words: list[str]) -> np.ndarray:
    """(N, 5) int8 array of letter indices 0..25, contiguous for fast access."""
    arr = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    arr = arr.reshape(len(words), WORD_LEN).astype(np.int8) - ord("a")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class Vocabulary:
    """The set of words the solver and model operate over."""

    words: list[str]
    index: dict[str, int]
    letters: np.ndarray  # (N, 5) int8 letter indices

    def __len__(self) -> int:
        return len(self.words)

    def encode(self, word: str) -> int:
        return self.index[word.lower()]

    def get(self, word: str) -> int | None:
        return self.index.get(word.lower())

    def decode(self, idx: int) -> str:
        return self.words[idx]

    @classmethod
    def from_words(cls, words: list[str]) -> "Vocabulary":
        index = {w: i for i, w in enumerate(words)}
        return cls(words=words, index=index, letters=words_to_letters(words))


def load_vocabulary(data_dir: str | Path = DATA_DIR) -> Vocabulary:
    """Load the official answer list as the working vocabulary."""
    return Vocabulary.from_words(load_word_list(Path(data_dir) / "answers.txt"))
