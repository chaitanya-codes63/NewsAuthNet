# src/models_tf.py
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore[reportMissingImports]

def build_keras_model(
    name: str,
    vocab_size: int = 30000,
    max_len: int = 300,
    embed_dim: int = 128,
    hidden_dim: int = 96,
    filters: int = 128,
    kernel_sizes: tuple = (3, 4, 5),
    dropout: float = 0.3,
    lr: float = 1e-3,
) -> tf.keras.Model:
    inp = layers.Input(shape=(max_len,), dtype="int32", name="token_ids")
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=False,
        name="embedding",
    )(inp)
    x = layers.SpatialDropout1D(dropout, name="embed_dropout")(x)

    if name == "lstm":
        seq = layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=True), name="bilstm"
        )(x)
        avg_pool = layers.GlobalAveragePooling1D()(seq)
        max_pool = layers.GlobalMaxPooling1D()(seq)
        feat = layers.Concatenate(name="mean_max_pool")([avg_pool, max_pool])

    elif name == "gru":
        seq = layers.Bidirectional(
            layers.GRU(hidden_dim, return_sequences=True), name="bigru"
        )(x)
        avg_pool = layers.GlobalAveragePooling1D()(seq)
        max_pool = layers.GlobalMaxPooling1D()(seq)
        feat = layers.Concatenate(name="mean_max_pool")([avg_pool, max_pool])

    elif name == "cnn":
        conv_pools = []
        for k in kernel_sizes:
            c = layers.Conv1D(
                filters=filters,
                kernel_size=k,
                padding="same",
                activation="relu",
                name=f"conv_k{k}",
            )(x)
            p = layers.GlobalMaxPooling1D(name=f"maxpool_k{k}")(c)
            conv_pools.append(p)
        feat = layers.Concatenate(name="concat_ngrams")(conv_pools)
    else:
        raise ValueError(f"Unknown model name: {name}")

    out = layers.Dropout(dropout, name="classifier_dropout")(feat)
    out = layers.Dense(1, activation="sigmoid", name="prob_real")(out)

    model = models.Model(inputs=inp, outputs=out, name=f"NewsAuth_{name.upper()}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model

def save_keras_pkl(model: tf.keras.Model, config: dict, path: str | Path):
    """Saves Keras weights (list of numpy arrays) + config dictionary into a .pkl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "config": config,
        "weights": model.get_weights(),
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

def load_keras_pkl(path: str | Path) -> tf.keras.Model:
    """Rebuilds the Keras model from config and loads weights from the .pkl file."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    cfg = bundle["config"]
    model = build_keras_model(**cfg)
    model.set_weights(bundle["weights"])
    return model