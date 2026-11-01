# src/explain.py
"""Explainability: follow ONE article through the model and show exactly how the verdict is formed.

    python -m src.explain --model cnn --text "paste article here"
    python -m src.explain --model lstm --file news1.txt --top 12
"""
from __future__ import annotations

import argparse
import html as _html

import numpy as np
import pandas as pd
import tensorflow as tf

from src.predict import LABEL_NAMES, Predictor
from src.preprocessing import clean_text

PAD_ID, UNK_ID = 0, 1
MAX_OCCLUSION_WORDS = 400


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0))))


def _verdict(p_real: float) -> tuple[int, str, float]:
    label = int(p_real >= 0.5)
    return label, LABEL_NAMES[label], p_real if label == 1 else 1.0 - p_real


def _logit_batch(net: tf.keras.Model, x_np: np.ndarray) -> np.ndarray:
    """Computes exact pre-sigmoid logit z = feats @ w + b for a batch of token IDs."""
    w_out, b_out = net.get_layer("prob_real").get_weights()
    w_vec, bias = w_out[:, 0], float(b_out[0])
    feats, _ = net.features(x_np)
    return feats @ w_vec + bias


def _occlusion_neural(net: tf.keras.Model, x: np.ndarray, n: int, base_z: float) -> np.ndarray:
    """Logit drop when word i is replaced by <unk> (id=1). Positive = word pushed toward Real."""
    rows = np.repeat(x, n, axis=0)
    for i in range(n):
        rows[i, i] = UNK_ID
    outs = []
    for start in range(0, n, 64):
        outs.append(_logit_batch(net, rows[start : start + 64]))
    return base_z - np.concatenate(outs)


def _grad_x_input_neural(net: tf.keras.Model, x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Computes gradient x input and embedding L2 norm on the Embedding output via tf.GradientTape."""
    emb_layer = net.get_layer("embedding")
    w_out, b_out = net.get_layer("prob_real").get_weights()
    w_tf = tf.constant(w_out, dtype=tf.float32)
    b_tf = tf.constant(b_out[0], dtype=tf.float32)
    x_tf = tf.constant(x, dtype=tf.int32)

    with tf.GradientTape() as tape:
        emb_out = emb_layer(x_tf)  # (1, T, E)
        tape.watch(emb_out)
        if net.kind == "cnn":
            pools = []
            for k in net.kernel_sizes:
                c = net.get_layer(f"conv_k{k}")(emb_out)
                pools.append(tf.reduce_max(c, axis=1))
            feats_tf = tf.concat(pools, axis=-1)
        else:
            rnn_name = "bilstm" if net.kind == "lstm" else "bigru"
            mask_tf = tf.not_equal(x_tf, PAD_ID)
            seq_out = net.get_layer(rnn_name)(emb_out, mask=mask_tf, training=False)
            m_f = tf.cast(tf.expand_dims(mask_tf, -1), tf.float32)
            avg_p = tf.reduce_sum(seq_out * m_f, axis=1) / tf.maximum(tf.reduce_sum(m_f, axis=1), 1.0)
            max_p = tf.reduce_max(seq_out, axis=1)
            feats_tf = tf.concat([avg_p, max_p], axis=-1)
        logit_tf = tf.squeeze(tf.matmul(feats_tf, w_tf), axis=-1) + b_tf

    grads = tape.gradient(logit_tf, emb_out)  # (1, T, E)
    emb_np = emb_out[0, :n, :].numpy()
    grad_np = grads[0, :n, :].numpy() if grads is not None else np.zeros_like(emb_np)
    grad_x_in = (grad_np * emb_np).sum(axis=-1)
    emb_norm = np.linalg.norm(emb_np, axis=-1)
    return grad_x_in, emb_norm


def _cnn_rows(net, words: list[str], feats: np.ndarray, w: np.ndarray, contrib: np.ndarray, pos: np.ndarray):
    rows, f = [], net.filters
    for g, k in enumerate(net.kernel_sizes):
        for j in range(f):
            idx = g * f + j
            start = int(pos[idx]) - k // 2
            lo, hi = max(start, 0), min(start + k, len(words))
            trig = " ".join(words[lo:hi]) if feats[idx] > 0 and hi > lo else "(did not fire: ReLU = 0)"
            rows.append(
                dict(
                    layer=f"Conv filter, window={k} words",
                    unit=f"filter #{j}",
                    activation=float(feats[idx]),
                    weight=float(w[idx]),
                    contribution=float(contrib[idx]),
                    triggered_by=trig,
                )
            )
    return rows


def _rnn_rows(net, words: list[str], feats: np.ndarray, w: np.ndarray, contrib: np.ndarray, pos: np.ndarray, n: int):
    d = len(feats) // 2
    h = net.hidden
    rows = []
    for idx in range(len(feats)):
        kind, u = ("average over all words", idx) if idx < d else ("strongest word", idx - d)
        direction = "forward" if (u < h or not net.bidirectional) else "backward"
        if idx < d:
            trig = "(mean over the whole text)"
        else:
            t = int(pos[u])
            trig = words[t] if t < n else "(pad)"
        rows.append(
            dict(
                layer=f"{net.kind.upper()} {direction} unit, {kind}",
                unit=f"hidden unit #{u}",
                activation=float(feats[idx]),
                weight=float(w[idx]),
                contribution=float(contrib[idx]),
                triggered_by=trig,
            )
        )
    return rows


def _explain_neural(p: Predictor, raw: str, top_k: int) -> dict:
    net, tok = p.net, p.tok
    clean = clean_text(raw, strip_sources=True)
    words = clean.split()[: tok.max_len]
    n = len(words)
    if n == 0:
        raise ValueError("After cleaning, no usable words remain in the text.")

    x = tok.encode_batch([clean])  # (1, 300)
    ids = x[0, :n].tolist()
    known = [i != UNK_ID and i != PAD_ID for i in ids]

    grad_x_input, emb_norm = _grad_x_input_neural(net, x, n)

    feats_batch, pos_batch = net.features(x)
    feats, pos = feats_batch[0], pos_batch[0]
    w_out, b_out = net.get_layer("prob_real").get_weights()
    w = w_out[:, 0]
    bias = float(b_out[0])
    contrib = feats * w
    recon = float(contrib.sum() + bias)
    z = recon

    if net.kind == "cnn":
        rows = _cnn_rows(net, words, feats, w, contrib, pos)
    else:
        rows = _rnn_rows(net, words, feats, w, contrib, pos, n)

    units = pd.DataFrame(rows)
    units["abs"] = units["contribution"].abs()
    units = units.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    units.insert(0, "rank", range(1, len(units) + 1))

    occ = _occlusion_neural(net, x, n, z)
    tokens = pd.DataFrame(
        {
            "position": range(n),
            "token": words,
            "token_id": ids,
            "in_vocab": known,
            "embedding_norm": emb_norm,
            "grad_x_input": grad_x_input,
            "occlusion": occ,
        }
    )
    vocab_actual = min(tok.vocab_size, len(tok.tokenizer.word_index) + 1)
    return dict(
        z=z,
        bias=bias,
        recon=recon,
        units=units,
        tokens=tokens,
        clean=clean,
        n_words=n,
        n_unk=int(n - sum(known)),
        n_feat=int(len(feats)),
        vocab=vocab_actual,
        max_len=tok.max_len,
    )


def _neural_steps(name: str, net, info: dict, raw: str, p_real: float) -> list[dict]:
    T, E = info["max_len"], net.embed_dim
    pad = T - info["n_words"]
    steps = [
        ("1. Raw text", f"{len(raw):,} characters received.", "string"),
        (
            "2. Cleaning",
            "lower-case, remove HTML/URLs/punctuation, strip publisher tags like '(Reuters)'. "
            f"Result: {info['n_words']} words.",
            f"{info['n_words']} words",
        ),
        (
            "3. Vocabulary lookup",
            f"each word -> integer id (vocabulary = {info['vocab']:,} words). "
            f"{info['n_unk']} word(s) unknown -> <unk> (id 1).",
            f"{info['n_words']} ids",
        ),
        (
            "4. Pad / truncate",
            f"fixed length {T}: {info['n_words']} real ids + {pad} padding ids (id 0, ignored by the model).",
            f"(1, {T})",
        ),
        (
            "5. Embedding layer",
            f"each id selects a learned {E}-number vector (row of a {info['vocab']:,} x {E} table).",
            f"(1, {T}, {E})",
        ),
    ]
    if net.kind == "cnn":
        k = ", ".join(map(str, net.kernel_sizes))
        F = net.filters
        steps += [
            (
                "6. Convolution layer",
                f"{len(net.kernel_sizes)} groups of {F} filters slide over the word vectors "
                f"(windows of {k} words). A filter is a learned phrase detector.",
                f"(1, {T}, {F}) per group",
            ),
            (
                "7. ReLU + global max-pool",
                "each filter keeps only its strongest match anywhere in the text (negative values become 0).",
                f"(1, {F * len(net.kernel_sizes)})",
            ),
        ]
    else:
        D = net.hidden * 2
        steps += [
            (
                f"6. Bi{net.kind.upper()} layer",
                f"reads the words in order (and in reverse); every step updates {net.hidden} memory units per direction.",
                f"(1, {T}, {D})",
            ),
            (
                "7. Pooling",
                f"for each of the {D} units: average over all words AND strongest value over all words.",
                f"(1, {2 * D})",
            ),
        ]
    steps += [
        (
            "8. Output neuron",
            f"ONE neuron: z = sum(activation x weight) + bias = {info['recon'] - info['bias']:+.4f} "
            f"{info['bias']:+.4f} = {info['recon']:+.4f}",
            "1 number (logit)",
        ),
        (
            "9. Sigmoid + decision",
            f"P(Real) = 1 / (1 + e^-z) = {p_real:.4f}. Rule: P(Real) >= 0.5 -> Real, otherwise Fake.",
            "probability",
        ),
    ]
    return [dict(stage=a, what_happens=b, output=c) for a, b, c in steps]


def _explain_baseline(p: Predictor, raw: str, top_k: int) -> dict:
    clean = clean_text(raw, strip_sources=True)
    if not clean:
        raise ValueError("After cleaning, no usable words remain in the text.")
    X = p.vec.transform([clean]).tocsr()
    coef, b = p.clf.coef_[0], float(p.clf.intercept_[0])
    names = p.vec.get_feature_names_out()
    idx, vals = X.indices, X.data
    contrib = coef[idx] * vals
    z = float(contrib.sum() + b)
    units = pd.DataFrame(
        {
            "layer": "TF-IDF feature (1- or 2-word phrase)",
            "unit": names[idx],
            "activation": vals,
            "weight": coef[idx],
            "contribution": contrib,
            "triggered_by": names[idx],
        }
    )
    units["abs"] = units["contribution"].abs()
    units = units.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    units.insert(0, "rank", range(1, len(units) + 1))

    words = clean.split()[:MAX_OCCLUSION_WORDS]
    occ = np.zeros(len(words))
    for i in range(len(words)):
        t = " ".join(words[:i] + words[i + 1 :])
        zi = float(p.clf.decision_function(p.vec.transform([t]))[0])
        occ[i] = z - zi
    vocab = set(names)
    tokens = pd.DataFrame(
        {
            "position": range(len(words)),
            "token": words,
            "token_id": None,
            "in_vocab": [w in vocab for w in words],
            "embedding_norm": np.nan,
            "grad_x_input": np.nan,
            "occlusion": occ,
        }
    )
    return dict(
        z=z,
        bias=b,
        recon=z,
        units=units,
        tokens=tokens,
        clean=clean,
        n_words=len(clean.split()),
        n_unk=int(sum(w not in vocab for w in clean.split())),
        n_feat=len(names),
        vocab=len(names),
    )


def _baseline_steps(info: dict, raw: str, p_real: float) -> list[dict]:
    steps = [
        ("1. Raw text", f"{len(raw):,} characters received.", "string"),
        (
            "2. Cleaning",
            f"lower-case, strip HTML/URLs/punctuation and publisher tags. {info['n_words']} words.",
            f"{info['n_words']} words",
        ),
        (
            "3. TF-IDF vectoriser",
            f"counts every 1- and 2-word phrase from a {info['vocab']:,}-phrase dictionary, "
            "down-weights phrases common in all articles (IDF), L2-normalises.",
            f"sparse vector of {info['vocab']:,} numbers",
        ),
        (
            "4. Logistic regression (no hidden layer)",
            "ONE weight per phrase, learned from training data. "
            f"z = sum(tfidf x weight) + bias = {info['recon']:+.4f}",
            "1 number (logit)",
        ),
        (
            "5. Sigmoid + decision",
            f"P(Real) = 1/(1+e^-z) = {p_real:.4f}. Rule: P(Real) >= 0.5 -> Real, otherwise Fake.",
            "probability",
        ),
    ]
    return [dict(stage=a, what_happens=b, output=c) for a, b, c in steps]


def explain(text: str, model: str = "cnn", top_k: int = 15, predictor: Predictor | None = None) -> dict:
    """Return a full, self-consistent explanation of one prediction."""
    if not text or len(str(text).strip()) < 20:
        raise ValueError("The text is too short. Paste at least 20 characters.")
    raw = str(text).strip()[:20000]
    p = predictor or Predictor(model)
    info = _explain_baseline(p, raw, top_k) if model == "baseline" else _explain_neural(p, raw, top_k)
    p_real = _sigmoid(info["z"])
    label, name, conf = _verdict(p_real)
    steps = _baseline_steps(info, raw, p_real) if model == "baseline" else _neural_steps(model, p.net, info, raw, p_real)
    u = info["units"]
    summary = {
        "units_total": int(len(u)),
        "units_pushing_real": int((u["contribution"] > 0).sum()),
        "units_pushing_fake": int((u["contribution"] < 0).sum()),
        "sum_push_real": float(u.loc[u["contribution"] > 0, "contribution"].sum()),
        "sum_push_fake": float(u.loc[u["contribution"] < 0, "contribution"].sum()),
        "bias": float(info["bias"]),
    }
    return dict(
        model=model,
        label=label,
        verdict=name,
        p_real=p_real,
        confidence=conf,
        logit=float(info["z"]),
        logit_from_parts=float(info["recon"]),
        summary=summary,
        steps=pd.DataFrame(steps),
        units=u,
        tokens=info["tokens"],
        cleaned_text=info["clean"],
        top_k=top_k,
    )


def explain_json(text: str, model: str = "cnn", top_k: int = 50, predictor: Predictor | None = None) -> dict:
    """JSON-serializable wrapper around explain() for the FastAPI + React UI."""
    r = explain(text, model=model, top_k=top_k, predictor=predictor)
    tok_df = r["tokens"].copy()
    tok_Records = []
    for _, row in tok_df.iterrows():
        tok_Records.append({
            "position": int(row["position"]),
            "token": str(row["token"]),
            "in_vocab": bool(row["in_vocab"]),
            "embedding_norm": None if pd.isna(row["embedding_norm"]) else round(float(row["embedding_norm"]), 4),
            "grad_x_input": None if pd.isna(row["grad_x_input"]) else round(float(row["grad_x_input"]), 4),
            "occlusion": round(float(row["occlusion"]), 4),
        })
    return {
        "model": r["model"],
        "label": r["label"],
        "verdict": r["verdict"],
        "p_real": round(r["p_real"], 4),
        "confidence": round(r["confidence"], 4),
        "logit": round(r["logit"], 6),
        "logit_from_parts": round(r["logit_from_parts"], 6),
        "logit_diff": float(abs(r["logit_from_parts"] - r["logit"])),
        "summary": r["summary"],
        "steps": r["steps"].to_dict(orient="records"),
        "units": r["units"].head(50).to_dict(orient="records"),
        "tokens": tok_Records,
        "cleaned_text": r["cleaned_text"],
    }


def highlight_html(tokens: pd.DataFrame, column: str = "occlusion") -> str:
    s = tokens[column].to_numpy(dtype=float)
    s = np.nan_to_num(s)
    m = max(np.abs(s).max(), 1e-9)
    parts = []
    for tok, v in zip(tokens["token"], s):
        a = min(abs(v) / m, 1.0) * 0.85
        color = f"rgba(41,128,185,{a:.2f})" if v > 0 else f"rgba(192,57,43,{a:.2f})"
        parts.append(
            f'<span title="{v:+.3f}" style="background:{color};padding:1px 3px;margin:1px;'
            f'border-radius:3px;display:inline-block">{_html.escape(str(tok))}</span>'
        )
    return " ".join(parts)


def _top_words(tokens: pd.DataFrame, k: int) -> str:
    t = tokens.reindex(tokens["occlusion"].abs().sort_values(ascending=False).index).head(k)
    cols = ["position", "token", "in_vocab", "occlusion"] + (
        ["grad_x_input"] if t["grad_x_input"].notna().any() else []
    )
    return t[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}")


def format_report(r: dict, top_k: int | None = None) -> str:
    k = top_k or r["top_k"]
    pd.set_option("display.width", 200, "display.max_colwidth", 48)
    s = r["summary"]
    out = [
        f"MODEL: {r['model']}",
        f"VERDICT: {r['verdict'].upper()}-like   (P(Real) = {r['p_real']:.4f}, confidence {r['confidence']:.1%})",
        "",
        "=== HOW THE TEXT FLOWS THROUGH THE MODEL ===",
        r["steps"].to_string(index=False),
        "",
        "=== THE FINAL DECISION (exact) ===",
        f"z = sum of {s['units_total']} neuron contributions + bias",
        f"  pushing toward REAL: {s['units_pushing_real']} units, total {s['sum_push_real']:+.3f}",
        f"  pushing toward FAKE: {s['units_pushing_fake']} units, total {s['sum_push_fake']:+.3f}",
        f"  bias: {s['bias']:+.3f}",
        f"  z = {r['logit_from_parts']:+.4f}   (model output: {r['logit']:+.4f})",
        f"  P(Real) = sigmoid(z) = {r['p_real']:.4f}  ->  {r['verdict']}",
        "",
        f"=== TOP {k} NEURONS / FEATURES BY |CONTRIBUTION| (positive = toward Real) ===",
        r["units"]
        .head(k)[["rank", "layer", "unit", "activation", "weight", "contribution", "triggered_by"]]
        .to_string(index=False, float_format=lambda v: f"{v:+.3f}"),
        "",
        "=== MOST INFLUENTIAL WORDS (occlusion: change in z when the word is removed) ===",
        _top_words(r["tokens"], k),
    ]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cnn", choices=["baseline", "lstm", "gru", "cnn"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    body = a.text if a.text else open(a.file, encoding="utf-8").read()
    try:
        print(format_report(explain(body, a.model, a.top), a.top))
    except Exception as e:
        raise SystemExit(f"ERROR: {e}")