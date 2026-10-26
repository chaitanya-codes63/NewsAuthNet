import { useState, useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState("lstm");
  const [installed, setInstalled] = useState(["baseline", "cnn", "lstm", "gru"]);
  const [metrics, setMetrics] = useState({});
  const [singleResult, setSingleResult] = useState(null);
  const [allResults, setAllResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/models`)
      .then((r) => r.json())
      .then((data) => {
        if (data.installed?.length) setInstalled(data.installed);
        if (data.metrics) setMetrics(data.metrics);
      })
      .catch(() => {});
  }, []);

  const handleAnalyze = async (compareAll = false) => {
    setError("");
    setSingleResult(null);
    setAllResults([]);
    setLoading(true);
    try {
      const endpoint = compareAll ? "/api/analyze_all" : "/api/analyze";
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: selectedModel }),
      });
      const data = await res.json();
      if (data.error) setError(data.error);
      else if (compareAll) setAllResults(data.results || []);
      else setSingleResult(data);
    } catch {
      setError("Cannot reach FastAPI backend on port 8000. Run: uvicorn src.api:app --port 8000");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Header */}
        <header className="border-b border-slate-800 pb-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-white">
              🛡️ NewsAuthNet: TensorFlow & Keras Fake News Detector
            </h1>
            <p className="text-sm text-slate-400">
              Leakage-controlled NLP pipeline (.pkl weights) — BiLSTM, BiGRU, TextCNN & TF-IDF Baseline
            </p>
          </div>
          <span className="text-xs bg-blue-900/50 border border-blue-600 text-blue-200 px-3 py-1 rounded-full">
            FastAPI + React UI
          </span>
        </header>

        {/* Disclaimer */}
        <div className="bg-amber-950/50 border border-amber-600/60 text-amber-200 px-4 py-3 rounded-lg text-sm">
          <strong>Disclaimer:</strong> Predictions reflect linguistic patterns learned from the ISOT dataset (Fake vs. Real) and do not constitute independent factual verification.
        </div>

        {/* Input Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap gap-4 items-center justify-between">
            <label className="text-sm font-semibold text-slate-300">
              Select Architecture:
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="ml-3 bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-1.5 text-sm"
              >
                {installed.map((m) => (
                  <option key={m} value={m}>
                    {m.toUpperCase()}
                  </option>
                ))}
              </select>
            </label>
            <span className="text-xs text-slate-400">{text.length.toLocaleString()} / 20,000 chars</span>
          </div>

          <textarea
            rows={6}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste a news article title and body here (minimum 20 characters)..."
            className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
          />

          <div className="flex gap-3">
            <button
              onClick={() => handleAnalyze(false)}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white font-semibold px-5 py-2.5 rounded-lg text-sm transition"
            >
              {loading ? "Analyzing..." : `Analyze with ${selectedModel.toUpperCase()}`}
            </button>
            <button
              onClick={() => handleAnalyze(true)}
              disabled={loading}
              className="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-semibold px-5 py-2.5 rounded-lg text-sm transition"
            >
              Compare All Models Side-by-Side
            </button>
          </div>

          {error && (
            <div className="bg-rose-950/70 border border-rose-600 text-rose-200 p-3 rounded-lg text-sm">
              {error}
            </div>
          )}

          {/* Single Model Result */}
          {singleResult && (
            <div
              className={`p-4 rounded-xl border ${
                singleResult.label === "Real"
                  ? "bg-emerald-950/50 border-emerald-600 text-emerald-100"
                  : "bg-rose-950/50 border-rose-600 text-rose-100"
              }`}
            >
              <div className="text-lg font-bold">
                {singleResult.label === "Real" ? "✅ Likely Authentic (Real-like)" : "🚨 Potential Misinformation (Fake-like)"}
              </div>
              <div className="text-sm mt-1">
                Model: <strong>{singleResult.model.toUpperCase()}</strong> | P(Real):{" "}
                <strong>{(singleResult.score_real * 100).toFixed(1)}%</strong> | Latency:{" "}
                <strong>{singleResult.latency_ms} ms</strong>
              </div>
            </div>
          )}

          {/* All Models Side-by-Side Comparison */}
          {allResults.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 pt-2">
              {allResults.map((r) => (
                <div
                  key={r.model}
                  className={`p-3.5 rounded-lg border ${
                    r.label === "Real"
                      ? "bg-emerald-950/40 border-emerald-600/70"
                      : "bg-rose-950/40 border-rose-600/70"
                  }`}
                >
                  <div className="text-xs text-slate-400 font-semibold">{r.model}</div>
                  <div className="text-base font-bold mt-1">
                    {r.label === "Real" ? "✅ Real" : "🚨 Fake"} ({r.confidence}%)
                  </div>
                  <div className="text-xs text-slate-400 mt-1">Latency: {r.latency_ms} ms</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Held-Out 15% Test Set Benchmark Table */}
        {Object.keys(metrics).length > 0 && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h2 className="text-base font-bold mb-3">📊 Held-Out 15% Test Split Performance</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2">Model</th>
                    <th className="py-2">Accuracy</th>
                    <th className="py-2">Macro-F1</th>
                    <th className="py-2">ROC-AUC</th>
                    <th className="py-2">Fake Recall</th>
                    <th className="py-2">Real Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics).map(([name, m]) => (
                    <tr key={name} className="border-b border-slate-800/60">
                      <td className="py-2 font-semibold uppercase">{name}</td>
                      <td className="py-2">{(m.accuracy * 100).toFixed(2)}%</td>
                      <td className="py-2">{(m.macro_f1 * 100).toFixed(2)}%</td>
                      <td className="py-2">{m.roc_auc ? m.roc_auc.toFixed(4) : "N/A"}</td>
                      <td className="py-2">{(m.per_class.Fake.recall * 100).toFixed(2)}%</td>
                      <td className="py-2">{(m.per_class.Real.recall * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}