# src/predict.py
import pickle
from dataclasses import dataclass
from pathlib import Path
from src.preprocessing import clean_text, TextTokenizer
from src.models_tf import load_keras_pkl

MODELS_DIR = Path("models")
AVAILABLE = ("baseline", "cnn", "lstm", "gru")
LABEL_NAMES = {0: "Fake", 1: "Real"}

@dataclass
class Prediction:
    label: int
    label_name: str
    score_real: float
    model: str

def installed_models() -> list[str]:
    return [m for m in AVAILABLE if (MODELS_DIR / m / "model.pkl").exists()]

class Predictor:
    def __init__(self, name: str):
        if name not in AVAILABLE:
            raise ValueError(f"Unknown model '{name}'. Choose from {AVAILABLE}")
        self.name = name
        self.dir = MODELS_DIR / name
        if not (self.dir / "model.pkl").exists():
            raise FileNotFoundError(
                f"Model '{name}' not found in {self.dir / 'model.pkl'}. Run step2_train.py first."
            )

        if name == "baseline":
            with open(self.dir / "model.pkl", "rb") as f:
                bundle = pickle.load(f)
            self.vec = bundle["vectorizer"]
            self.clf = bundle["model"]
        else:
            self.tok = TextTokenizer.load(self.dir / "tokenizer.pkl")
            self.net = load_keras_pkl(self.dir / "model.pkl")

    def predict(self, raw_text: str) -> Prediction:
        if not raw_text or len(str(raw_text).strip()) < 20:
            raise ValueError("Please provide at least 20 characters of article text.")
        cleaned = clean_text(str(raw_text)[:20000], strip_sources=True)
        if not cleaned:
            raise ValueError("No valid words remained after text cleaning.")

        if self.name == "baseline":
            X = self.vec.transform([cleaned])
            score_real = float(self.clf.predict_proba(X)[0, 1])
        else:
            X = self.tok.encode_batch([cleaned])
            score_real = float(self.net(X, training=False).numpy()[0, 0])

        label = int(score_real >= 0.5)
        return Prediction(
            label=label,
            label_name=LABEL_NAMES[label],
            score_real=score_real,
            model=self.name,
        )

    def predict_batch(self, raw_texts: list[str]) -> list[float]:
        """Fast batch inference on the 15% test split."""
        cleaned = [clean_text(str(t), strip_sources=True) for t in raw_texts]
        if self.name == "baseline":
            X = self.vec.transform(cleaned)
            return self.clf.predict_proba(X)[:, 1].tolist()
        else:
            X = self.tok.encode_batch(cleaned)
            preds = self.net.predict(X, batch_size=128, verbose=1)
            return preds[:, 0].tolist()