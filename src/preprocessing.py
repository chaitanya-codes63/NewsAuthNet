import re
import pickle
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer  # type: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore[reportMissingImports]

_URL = re.compile(r"https?://\S+|www\.\S+")
_TAG = re.compile(r"<[^>]+>")
_NON_WORD = re.compile(r"[^a-z0-9\s']")
_SPACES = re.compile(r"\s+")
_SOURCE_TAG = re.compile(
    r"^\s*(?:[A-Z][A-Za-z .,/&'-]{0,40}\s)?\((?:Reuters|AP|AFP)\)\s*[-–—]*\s*",
    re.IGNORECASE,
)
_SOURCE_WORDS = re.compile(
    r"\b(?:reuters|21st century wire|featured image via\b.*)", re.IGNORECASE
)

def strip_source_tags(text: str) -> str:
    text = _SOURCE_TAG.sub("", text)
    return _SOURCE_WORDS.sub(" ", text)

def clean_text(text: str, strip_sources: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    if strip_sources:
        text = strip_source_tags(text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()

class TextTokenizer:
    """Wraps Keras Tokenizer and saves/loads strictly as a .pkl file."""
    def __init__(self, vocab_size: int = 30000, max_len: int = 300):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = KerasTokenizer(num_words=vocab_size, oov_token="<unk>")

    def fit(self, texts: list[str]):
        self.tokenizer.fit_on_texts(texts)
        return self

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            seqs, maxlen=self.max_len, padding="post", truncating="post", value=0
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vocab_size": self.vocab_size,
                    "max_len": self.max_len,
                    "tokenizer": self.tokenizer,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "TextTokenizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(vocab_size=data["vocab_size"], max_len=data["max_len"])
        obj.tokenizer = data["tokenizer"]
        return obj