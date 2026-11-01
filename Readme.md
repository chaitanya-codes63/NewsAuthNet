# NewsAuthNet: Fake News Detection Using Deep Learning and NLP

**NewsAuthNet** is a leakage-controlled, full-stack text classification system that detects misinformation in English news articles. Developed as a final-year CS(AI) course project, it compares a TF-IDF baseline against three deep learning architectures (BiLSTM, BiGRU, and Text-CNN) trained on a rigorously audited version of the **ISOT Fake News Dataset**. 

Rather than stopping at offline Jupyter Notebooks, the project deploys these trained models through a **FastAPI** microservice, a modern **React + Tailwind** web dashboard with neuron-level explainability, and a **Manifest V3 Chrome Extension** for real-time, in-browser credibility analysis.

> **Disclaimer:** The system predicts dataset labels (Real-like / Fake-like) based on linguistic style and editorial patterns learned during training. It does *not* independently verify factual claims.

---

## 🌟 Key Features

*   **Leakage-Free Data Pipeline:** Deterministic deduplication, conflicting label resolution, and regex-based stripping of publisher shortcuts (e.g., `(Reuters)`) prior to 70/15/15 stratified splitting.
*   **Four-Model Benchmark (TensorFlow/Keras):**
    *   `Baseline`: TF-IDF (1-2 grams, 50k features) + Logistic Regression.
    *   `BiLSTM`: Bidirectional LSTM with dual mean/max pooling.
    *   `BiGRU`: Bidirectional GRU (Highest test-set performance: **99.15% Macro-F1**).
    *   `Text-CNN`: Parallel 1D Convolutions (k=3, 4, 5).
*   **Neuron-Level Explainability:** A custom white-box explainer (`src/explain.py`) traces predictions layer-by-layer, computing exact penultimate neuron contributions ($activation \times weight$) and word-level occlusion heatmaps.
*   **Dual Deployment Architecture:**
    *   **React Dashboard (`frontend/`)**: Interactive UI for 4-model side-by-side comparison, test-set benchmark telemetry, and visual XAI traces.
    *   **Chrome Extension (`extension/`)**: 1-click DOM scraping of live news portals with sub-100ms in-browser alerts.
*   **`.pkl` Artifact Persistence:** All TensorFlow weights, hyperparameter configurations, and Keras Tokenizers are serialized into portable `.pkl` bundles.

---

## 🛠️ Project Layout

```text
FakeNews_TF_React/
├── configs/
│   └── config.yaml              # Centralized hyperparameters & paths
├── data/
│   ├── raw/                     # Fake.csv, True.csv (You provide these)
│   └── processed/               # Cleaned train.csv, val.csv, test.csv
├── extension/                   # Manifest V3 Chrome Extension
│   ├── manifest.json
│   ├── popup.html
│   └── popup.js
├── frontend/                    # React 18 + Vite + Tailwind UI
│   ├── src/App.jsx              # Main Dashboard
│   └── package.json
├── models/                      # Serialized .pkl models and tokenizers
│   ├── baseline/
│   ├── cnn/
│   ├── gru/
│   └── lstm/
├── reports/                     # Auto-generated metrics & audit logs
│   ├── figures/                 # Confusion matrices, learning curves
│   ├── data_audit.json
│   └── metrics.json
├── src/                         # Core Python Pipeline
│   ├── step1_eda_and_split.py   # Data auditing & leakage control
│   ├── preprocessing.py         # Regex cleaning & Keras Tokenizer
│   ├── models_tf.py             # Keras architectures & .pkl saving
│   ├── step2_train.py           # EpochLogger & model training
│   ├── step3_evaluate.py        # Test split evaluation
│   ├── predict.py               # Unified Predictor interface
│   ├── explain.py               # Neuron & Word-level XAI engine
│   └── api.py                   # FastAPI microservice
├── requirements.txt
└── README.md
```

---

## 🚀 Setup and Installation

### 1. Create the Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```
*(Ensure TensorFlow 2.15+ and Scikit-Learn 1.3+ are installed)*

### 2. Prepare the Dataset
Download the **ISOT Fake News Dataset** and place `Fake.csv` and `True.csv` inside `data/raw/`.

---

## 🧠 Running the Pipeline

### Step 1: Audit, Clean, and Split Data
This script strips `(Reuters)` tags, drops duplicates, and creates the `data/processed/` splits.
```bash
python -m src.step1_eda_and_split
```

### Step 2: Train the Models
Train the baseline and neural models. The custom `EpochLogger` will display live `val_loss` and `accuracy`.
```bash
python -m src.step2_train --models baseline cnn lstm gru --epochs 6 --batch-size 64
```
*(Note: To train faster on a CPU, use `--batch-size 128` or train `cnn` separately first).*

### Step 3: Evaluate on Held-Out Test Split
Generates benchmark metrics, confusion matrices, and learning curves in `reports/`.
```bash
python -m src.step3_evaluate
```

### Step 4: Terminal CLI Explainability (Optional)
Trace a prediction exactly via the command line:
```bash
# Assuming you have a sample text file named news1.txt in the root
python -m src.explain --model lstm --file news1.txt --top 10
```

---

## 🌐 Starting the Web Services

### 1. Start the FastAPI Backend
Pre-loads all `.pkl` models into system RAM for sub-100ms inference.
```bash
uvicorn src.api:app --reload --port 8000
```

### 2. Start the React Frontend Dashboard
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to access the interactive dashboard.

### 3. Load the Chrome Extension
1. Open Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (top right toggle).
3. Click **Load unpacked** and select the `extension/` folder from this repository.
4. Pin the **News Sentinel** extension. Click it on any live news article while the FastAPI server is running!

---

## 📊 Evaluation Results (15% Test Split)

Evaluated on 5,820 unseen, deduplicated, and tag-stripped articles:

| Architecture | Accuracy | Macro-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **BiGRU** | **99.16%** | **99.15%** | **0.9994** |
| **BiLSTM** | 99.14% | 99.14% | 0.9993 |
| **TF-IDF + LR** | 98.76% | 98.76% | 0.9989 |
| **Text-CNN** | 98.49% | 98.48% | 0.9992 |

*Note: The recurrent models (BiLSTM/BiGRU) successfully reduced the baseline error rate by over 30% by capturing sequential, bidirectional journalistic context.*