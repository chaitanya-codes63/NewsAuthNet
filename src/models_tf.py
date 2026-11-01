# src/models_tf.py
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def _attach_features_method(model: tf.keras.Model, cfg: dict) -> tf.keras.Model:
    """Attaches .features(x) to the Keras model so explain.py can read exact penultimate activations."""
    name = cfg["name"]
    model.kind = name
    model.hidden = cfg.get("hidden_dim", 96)
    model.bidirectional = True
    model.filters = cfg.get("filters", 128)
    model.kernel_sizes = tuple(cfg.get("kernel_sizes", (3, 4, 5)))
    model.embed_dim = cfg.get("embed_dim", 128)

    if name == "cnn":
        conv_outs = [model.get_layer(f"conv_k{k}").output for k in model.kernel_sizes]
        pooled_out = model.get_layer("concat_ngrams").output
        probe = tf.keras.Model(inputs=model.inputs, outputs=conv_outs + [pooled_out])

        def features(x_np: np.ndarray):
            outs = [t.numpy() for t in probe(x_np, training=False)]
            conv_maps, pooled = outs[:-1], outs[-1]
            mask = (x_np != 0)[:, :, None]  # (B, T, 1)
            pos_list = []
            for c_map in conv_maps:
                masked_c = np.where(mask, c_map, -1e9)
                pos_list.append(np.argmax(masked_c, axis=1))  # (B, F)
            return pooled, np.concatenate(pos_list, axis=1)

        model.features = features
    else:
        rnn_name = "bilstm" if name == "lstm" else "bigru"
        seq_out = model.get_layer(rnn_name).output
        pooled_out = model.get_layer("mean_max_pool").output
        probe = tf.keras.Model(inputs=model.inputs, outputs=[seq_out, pooled_out])

        def features(x_np: np.ndarray):
            seq_val, pooled = [t.numpy() for t in probe(x_np, training=False)]
            mask = (x_np != 0)[:, :, None]  # (B, T, 1)
            masked_seq = np.where(mask, seq_val, -1e9)
            max_pos = np.argmax(masked_seq, axis=1)  # (B, 2*H)
            return pooled, max_pos

        model.features = features

    return model


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
        mask_zero=(name in ("lstm", "gru")),
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
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    cfg = dict(
        name=name, vocab_size=vocab_size, max_len=max_len, embed_dim=embed_dim,
        hidden_dim=hidden_dim, filters=filters, kernel_sizes=kernel_sizes,
        dropout=dropout, lr=lr,
    )
    return _attach_features_method(model, cfg)


def save_keras_pkl(model: tf.keras.Model, config: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"config": config, "weights": model.get_weights()}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_keras_pkl(path: str | Path) -> tf.keras.Model:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    cfg = bundle["config"]
    model = build_keras_model(**cfg)
    model.set_weights(bundle["weights"])
    return model