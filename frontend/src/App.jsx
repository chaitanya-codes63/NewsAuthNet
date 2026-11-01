import { useState, useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000";

const MODEL_META = {
  lstm: {
    title: "BiLSTM",
    nice: "BiLSTM",
    badge: "Recurrent • 384-d Pool",
    desc: "Captures long-range forward & backward sequential context.",
  },
  cnn: {
    title: "TextCNN",
    nice: "Text CNN",
    badge: "Convolutional • N-Grams",
    desc: "Detects local phrase patterns & sensational n-gram triggers.",
  },
  gru: {
    title: "BiGRU",
    nice: "GRU",
    badge: "Fast Recurrent • 2-Gate",
    desc: "Streamlined recurrent gating with lower parameter overhead.",
  },
  baseline: {
    title: "TF-IDF + LR",
    nice: "TF-IDF + Logistic Regression",
    badge: "Classical ML • 50k Features",
    desc: "Unigram & bigram frequency baseline tuned on validation F1.",
  },
};

const SAMPLE_ARTICLES = [
  {
    label: "Load news1.txt (Real)",
    text: "The central bank announced on Thursday that benchmark interest rates would remain unchanged following a two-day monetary policy review. Officials cited moderating inflation across food and energy sectors while projecting steady gross domestic product growth over the upcoming fiscal quarter. Economists surveyed ahead of the decision largely anticipated the hold as policymakers balance price stability with industrial credit demand.",
  },
  {
    label: "Load news1-fake.txt (Fake)",
    text: "SHOCKING: Secret leaked documents prove that global weather satellites are being used to mind-control voters using invisible frequency beams! Mainstream media refuses to air this explosive truth after whistleblowers revealed that every smartphone battery is secretly recording your thoughts and sending them to an underground bunker. Share this immediately before they delete the internet!",
  },
];

export default function App() {
  const [mainTab, setMainTab] = useState("predict"); // "predict" | "comparison"
  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState("cnn");
  const [installed, setInstalled] = useState(["cnn", "lstm", "gru", "baseline"]);
  const [metrics, setMetrics] = useState({});
  const [singleResult, setSingleResult] = useState(null);
  const [allResults, setAllResults] = useState([]);
  const [explanation, setExplanation] = useState(null);

  // Explainability sub-states matching friend's app.py
  const [activeExplainTab, setActiveExplainTab] = useState("pipeline");
  const [showCleanedText, setShowCleanedText] = useState(false);
  const [topN, setTopN] = useState(15);
  const [wordMethod, setWordMethod] = useState("occlusion"); // "occlusion" | "grad_x_input"

  // Model comparison tab states matching friend's app.py
  const [chartMetric, setChartMetric] = useState("Macro F1");
  const [confusionModel, setConfusionModel] = useState("gru");

  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");

  useEffect(() => {
    const root = document.documentElement;
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = () => {
      const isDark = theme === "dark" || (theme === "system" && mediaQuery.matches);
      root.classList.toggle("dark", isDark);
    };
    applyTheme();
    localStorage.setItem("theme", theme);
    mediaQuery.addEventListener("change", applyTheme);
    return () => mediaQuery.removeEventListener("change", applyTheme);
  }, [theme]);

  useEffect(() => {
    fetch(`${API_BASE}/api/models`)
      .then((r) => r.json())
      .then((data) => {
        setApiOnline(true);
        if (data.installed?.length) setInstalled(data.installed);
        if (data.metrics) {
          setMetrics(data.metrics);
          const keys = Object.keys(data.metrics);
          if (keys.length && !data.metrics[confusionModel]) setConfusionModel(keys[0]);
        }
      })
      .catch(() => setApiOnline(false));
  }, []);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const handleAnalyze = async (compareAll = false) => {
    setError("");
    setSingleResult(null);
    setAllResults([]);
    setExplanation(null);
    setLoading(true);
    try {
      if (compareAll) {
        const res = await fetch(`${API_BASE}/api/analyze_all`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, model: selectedModel }),
        });
        const data = await res.json();
        if (data.error) setError(data.error);
        else setAllResults(data.results || []);
      } else {
        const [resPred, resExp] = await Promise.all([
          fetch(`${API_BASE}/api/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, model: selectedModel }),
          }),
          fetch(`${API_BASE}/api/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, model: selectedModel }),
          }),
        ]);
        const dataPred = await resPred.json();
        const dataExp = await resExp.json();
        if (dataPred.error) setError(dataPred.error);
        else {
          setSingleResult(dataPred);
          if (!dataExp.error) {
            setExplanation(dataExp);
            if (selectedModel === "baseline" && wordMethod === "grad_x_input") {
              setWordMethod("occlusion");
            }
          }
        }
      }
      setApiOnline(true);
    } catch {
      setApiOnline(false);
      setError("FastAPI backend is offline. Start it with: uvicorn src.api:app --port 8000");
    } finally {
      setLoading(false);
    }
  };

  // Prepare tokens for Tab 3 (Occlusion vs Gradient x Input)
  const effectiveMethod =
    explanation?.model === "baseline" && wordMethod === "grad_x_input"
      ? "occlusion"
      : wordMethod;

  const maxTokenAbs = explanation?.tokens?.length
    ? Math.max(...explanation.tokens.map((t) => Math.abs(t[effectiveMethod] || 0)), 1e-9)
    : 1;

  const top12Words = explanation?.tokens
    ? [...explanation.tokens]
        .sort((a, b) => Math.abs(b[effectiveMethod] || 0) - Math.abs(a[effectiveMethod] || 0))
        .slice(0, 12)
    : [];

  const slicedUnits = explanation?.units ? explanation.units.slice(0, topN) : [];
  const maxUnitContrib = slicedUnits.length
    ? Math.max(...slicedUnits.map((u) => Math.abs(u.contribution)), 1e-9)
    : 1;

  return (
    <div className="min-h-screen w-full bg-slate-50 dark:bg-[#090D16] text-slate-900 dark:text-slate-100 transition-colors duration-300">
      {/* Top Navbar */}
      <nav className="sticky top-0 z-30 border-b border-slate-200/80 dark:border-slate-800/80 bg-white/80 dark:bg-[#090D16]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-md shadow-blue-500/20">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-slate-900 dark:text-white">
                NewAuthNet: Fake News Detector
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400 hidden sm:block">
                Deep Learning & NLP course project • Predicts whether text resembles Real or Fake training articles
              </p>
            </div>
          </div>

          {/* Main View Switcher: Predict & Explain vs Model Comparison */}
          <div className="flex items-center gap-3">
            <div className="inline-flex rounded-xl p-1 bg-slate-200/70 dark:bg-slate-900 border border-slate-300/60 dark:border-slate-800">
              <button
                onClick={() => setMainTab("predict")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                  mainTab === "predict"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                }`}
              >
                Predict & Explain
              </button>
              <button
                onClick={() => setMainTab("comparison")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                  mainTab === "comparison"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                }`}
              >
                Model Comparison
              </button>
            </div>

            {/* Light / Dark / Auto Toggler */}
            <div className="inline-flex items-center rounded-xl p-1 bg-slate-200/70 dark:bg-slate-900 border border-slate-300/60 dark:border-slate-800">
              {[
                { id: "light", label: "Light" },
                { id: "dark", label: "Dark" },
                { id: "system", label: "Auto" },
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => setTheme(item.id)}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    theme === item.id
                      ? "bg-white dark:bg-slate-800 text-blue-600 dark:text-blue-400 shadow-sm"
                      : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* =====================================================================
            MAIN TAB 1: PREDICT & EXPLAIN
           ===================================================================== */}
        {mainTab === "predict" && (
          <>
            {/* 1. Architecture Selector Cards */}
            <section>
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  1. Select Model Architecture
                </h2>
                <span className="text-xs text-slate-400 dark:text-slate-500">
                  This is a machine-learning estimate based on training data, not independent fact verification.
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
                {["cnn", "lstm", "gru", "baseline"].map((key) => {
                  const info = MODEL_META[key];
                  const isSelected = selectedModel === key;
                  const modelMetric = metrics[key];

                  return (
                    <button
                      key={key}
                      onClick={() => setSelectedModel(key)}
                      className={`text-left p-4 rounded-2xl border transition-all duration-200 ${
                        isSelected
                          ? "bg-blue-50/70 dark:bg-blue-950/30 border-blue-500 ring-2 ring-blue-500/20 shadow-md"
                          : "bg-white dark:bg-slate-900/80 border-slate-200/90 dark:border-slate-800/90 hover:border-slate-300 dark:hover:border-slate-700"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-base font-bold text-slate-900 dark:text-white">{info.title}</span>
                        {modelMetric && (
                          <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded-md bg-emerald-100 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300">
                            F1: {(modelMetric.macro_f1 * 100).toFixed(2)}%
                          </span>
                        )}
                      </div>
                      <div className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-1.5">{info.badge}</div>
                      <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">{info.desc}</p>
                    </button>
                  );
                })}
              </div>
            </section>

            {/* 2. Input & Verdict Workspace */}
            <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              <div className="lg:col-span-7 bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 rounded-2xl shadow-sm overflow-hidden">
                <div className="px-5 py-3.5 border-b border-slate-200/80 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2 bg-slate-50/60 dark:bg-slate-900/50">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    2. Article Title and Body
                  </span>
                  <div className="flex flex-wrap items-center gap-2">
                    {SAMPLE_ARTICLES.map((s, idx) => (
                      <button
                        key={idx}
                        onClick={() => setText(s.text)}
                        className="text-xs font-medium px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 transition"
                      >
                        {s.label}
                      </button>
                    ))}
                    {text && (
                      <button
                        onClick={() => {
                          setText("");
                          setSingleResult(null);
                          setAllResults([]);
                          setExplanation(null);
                        }}
                        className="text-xs font-medium px-2.5 py-1 rounded-lg text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </div>

                <div className="p-5">
                  <textarea
                    rows={7}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Paste the article text here..."
                    className="w-full bg-slate-50/70 dark:bg-[#0B0F19] border border-slate-200 dark:border-slate-800 rounded-xl p-4 text-sm leading-relaxed text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/40 transition"
                  />
                  <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
                    <div className="text-xs font-mono text-slate-500 dark:text-slate-400">
                      {wordCount} words • {text.length.toLocaleString()} / 20,000 characters
                    </div>
                    <div className="flex flex-wrap items-center gap-2.5">
                      <button
                        onClick={() => handleAnalyze(true)}
                        disabled={loading}
                        className="px-4 py-2.5 rounded-xl text-xs font-semibold border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 transition"
                      >
                        Compare All 4 Models
                      </button>
                      <button
                        onClick={() => handleAnalyze(false)}
                        disabled={loading}
                        className="px-5 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-md shadow-blue-500/20 transition"
                      >
                        {loading ? "Analysing & Tracing Network..." : `Analyse (${MODEL_META[selectedModel]?.title})`}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Verdict Panel */}
              <div className="lg:col-span-5 bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 rounded-2xl p-5 shadow-sm min-h-[290px] flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-slate-200/80 dark:border-slate-800 pb-3 mb-4">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      3. Predicted Dataset Label
                    </span>
                    <span className="text-xs font-mono text-slate-400">Rule: P(Real) ≥ 0.50</span>
                  </div>

                  {error && (
                    <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/50 border border-rose-300 dark:border-rose-800 text-rose-700 dark:text-rose-200 text-xs">
                      {error}
                    </div>
                  )}

                  {!error && !singleResult && allResults.length === 0 && (
                    <div className="py-10 text-center text-xs text-slate-400">
                      Click <b>Analyse</b> to run the model and trace how each neuron contributed to the final logit.
                    </div>
                  )}

                  {singleResult && (
                    <div className="space-y-4">
                      <div
                        className={`p-4 rounded-xl border ${
                          singleResult.label === "Real"
                            ? "bg-emerald-50/70 dark:bg-emerald-950/30 border-emerald-400/70 dark:border-emerald-700"
                            : "bg-rose-50/70 dark:bg-rose-950/30 border-rose-400/70 dark:border-rose-700"
                        }`}
                      >
                        <div className="flex items-center justify-between text-xs font-mono opacity-80">
                          <span>{MODEL_META[singleResult.model]?.nice || singleResult.model.toUpperCase()}</span>
                          <span>{singleResult.latency_ms} ms</span>
                        </div>
                        <div className="text-xl font-extrabold mt-1">
                          Predicted dataset label: {singleResult.label === "Real" ? "Real-like" : "Fake-like"}
                        </div>
                        <div className="text-xs mt-1 font-mono">
                          Model confidence:{" "}
                          {(
                            (singleResult.label === "Real"
                              ? singleResult.score_real
                              : 1 - singleResult.score_real) * 100
                          ).toFixed(1)}
                          % (P(Real) = {singleResult.score_real.toFixed(3)})
                        </div>
                      </div>

                      <div className="space-y-1.5">
                        <div className="flex justify-between text-xs font-medium">
                          <span className="text-rose-500">0% (Fake)</span>
                          <span className="font-mono font-bold">Model confidence</span>
                          <span className="text-emerald-500">100% (Real)</span>
                        </div>
                        <div className="w-full h-3 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 rounded-full ${
                              singleResult.label === "Real"
                                ? "bg-gradient-to-r from-emerald-500 to-teal-400"
                                : "bg-gradient-to-r from-rose-600 to-orange-500"
                            }`}
                            style={{
                              width: `${Math.max(
                                4,
                                (singleResult.label === "Real"
                                  ? singleResult.score_real
                                  : 1 - singleResult.score_real) * 100
                              )}%`,
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {allResults.length > 0 && (
                    <div className="space-y-2">
                      {allResults.map((r) => (
                        <div key={r.model} className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 flex justify-between items-center">
                          <span className="text-xs font-bold">{r.model} ({r.latency_ms} ms)</span>
                          <span className={`text-xs font-bold px-2.5 py-1 rounded-lg ${r.label === "Real" ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"}`}>
                            {r.label === "Real" ? "Real-like" : "Fake-like"} • {r.confidence}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="mt-4 pt-3 border-t border-slate-200/70 dark:border-slate-800 text-[11px] text-slate-400">
                  This is a machine-learning estimate based on training data, not independent fact verification. The score is model confidence, not a calibrated measure of truth.
                </div>
              </div>
            </section>

            {/* =====================================================================
                3. HOW THE VERDICT WAS DECIDED (EXACT 4 TABS FROM FRIEND'S APP.PY)
               ===================================================================== */}
            {explanation && (
              <section className="bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-4">
                  <div>
                    <h2 className="text-lg font-bold text-slate-900 dark:text-white">
                      How the verdict was decided
                    </h2>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      Traced through {MODEL_META[explanation.model]?.nice || explanation.model.toUpperCase()} ({explanation.latency_ms} ms)
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-1.5 bg-slate-100 dark:bg-slate-950 p-1.5 rounded-xl border border-slate-200 dark:border-slate-800">
                    {[
                      { id: "pipeline", label: "1 · Pipeline" },
                      { id: "neurons", label: "2 · Which neurons decided" },
                      { id: "words", label: "3 · Which words mattered" },
                      { id: "sum", label: "4 · The final sum" },
                    ].map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveExplainTab(tab.id)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                          activeExplainTab === tab.id
                            ? "bg-blue-600 text-white shadow-sm"
                            : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                        }`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* TAB 1: PIPELINE */}
                {activeExplainTab === "pipeline" && (
                  <div className="space-y-4">
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      Your text, transformed step by step. The numbers are from the real trained model.
                    </p>
                    <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
                      <table className="w-full text-left text-xs sm:text-sm border-collapse">
                        <thead>
                          <tr className="bg-slate-100 dark:bg-slate-800/70 text-slate-500 dark:text-slate-300 border-b border-slate-200 dark:border-slate-800">
                            <th className="py-2.5 px-4">stage</th>
                            <th className="py-2.5 px-4">what_happens</th>
                            <th className="py-2.5 px-4">output</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                          {explanation.steps.map((s, idx) => (
                            <tr key={idx}>
                              <td className="py-2.5 px-4 font-semibold whitespace-nowrap">{s.stage}</td>
                              <td className="py-2.5 px-4 text-slate-600 dark:text-slate-300">{s.what_happens}</td>
                              <td className="py-2.5 px-4 font-mono text-blue-600 dark:text-blue-400 whitespace-nowrap">{s.output}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Expander: Text exactly as the model sees it */}
                    <div className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden">
                      <button
                        onClick={() => setShowCleanedText(!showCleanedText)}
                        className="w-full px-4 py-2.5 text-left text-xs font-semibold bg-slate-50 dark:bg-slate-950 flex justify-between items-center"
                      >
                        <span>Text exactly as the model sees it (after cleaning)</span>
                        <span>{showCleanedText ? "▲" : "▼"}</span>
                      </button>
                      {showCleanedText && (
                        <div className="p-4 text-xs font-mono text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-900 leading-relaxed">
                          {explanation.cleaned_text}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* TAB 2: WHICH NEURONS DECIDED */}
                {activeExplainTab === "neurons" && (
                  <div className="space-y-5">
                    <div className="flex flex-wrap items-center justify-between gap-4">
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        Each row is one neuron (or TF-IDF feature) that feeds the single output neuron.{" "}
                        <strong>contribution = activation × weight.</strong> Positive pushes toward REAL, negative toward FAKE.
                      </p>
                      <div className="flex items-center gap-3 text-xs">
                        <label className="font-semibold">Show top N: {topN}</label>
                        <input
                          type="range"
                          min={5}
                          max={50}
                          value={topN}
                          onChange={(e) => setTopN(Number(e.target.value))}
                          className="w-32 accent-blue-600"
                        />
                      </div>
                    </div>

                    {/* Units Table */}
                    <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl max-h-80">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead className="sticky top-0 bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-300">
                          <tr>
                            <th className="py-2 px-3">rank</th>
                            <th className="py-2 px-3">layer</th>
                            <th className="py-2 px-3">unit</th>
                            <th className="py-2 px-3">activation</th>
                            <th className="py-2 px-3">weight</th>
                            <th className="py-2 px-3">contribution</th>
                            <th className="py-2 px-3">triggered_by</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
                          {slicedUnits.map((u) => (
                            <tr key={u.rank}>
                              <td className="py-2 px-3">{u.rank}</td>
                              <td className="py-2 px-3 font-sans text-slate-500 dark:text-slate-400">{u.layer}</td>
                              <td className="py-2 px-3 font-semibold">{u.unit}</td>
                              <td className="py-2 px-3">{u.activation >= 0 ? `+${u.activation.toFixed(3)}` : u.activation.toFixed(3)}</td>
                              <td className="py-2 px-3">{u.weight >= 0 ? `+${u.weight.toFixed(3)}` : u.weight.toFixed(3)}</td>
                              <td className={`py-2 px-3 font-bold ${u.contribution >= 0 ? "text-blue-500" : "text-rose-500"}`}>
                                {u.contribution >= 0 ? `+${u.contribution.toFixed(3)}` : u.contribution.toFixed(3)}
                              </td>
                              <td className="py-2 px-3 font-sans text-slate-700 dark:text-slate-200">{u.triggered_by}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Horizontal Bar Chart of Neuron Contributions */}
                    <div className="space-y-2 pt-2">
                      <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                        Neuron Contribution Chart (unit · triggered_by)
                      </div>
                      <div className="space-y-1.5">
                        {slicedUnits.map((u) => {
                          const pct = Math.max(3, (Math.abs(u.contribution) / maxUnitContrib) * 100);
                          const pos = u.contribution >= 0;
                          return (
                            <div key={u.rank} className="grid grid-cols-12 items-center gap-2 text-xs font-mono">
                              <div className="col-span-5 sm:col-span-4 truncate text-slate-600 dark:text-slate-300">
                                {u.unit} · {String(u.triggered_by).slice(0, 28)}
                              </div>
                              <div className="col-span-5 sm:col-span-6 bg-slate-100 dark:bg-slate-800 h-2.5 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${pos ? "bg-blue-500" : "bg-rose-500"}`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                              <div className={`col-span-2 text-right font-bold ${pos ? "text-blue-500" : "text-rose-500"}`}>
                                {pos ? `+${u.contribution.toFixed(3)}` : u.contribution.toFixed(3)}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      {explanation.summary.units_total} units in total: {explanation.summary.units_pushing_real} push toward Real,{" "}
                      {explanation.summary.units_pushing_fake} toward Fake.
                    </p>
                  </div>
                )}

                {/* TAB 3: WHICH WORDS MATTERED */}
                {activeExplainTab === "words" && (
                  <div className="space-y-4">
                    {/* Radio Selector: Occlusion vs Gradient x input */}
                    <div className="flex flex-wrap items-center gap-4 text-xs">
                      <span className="font-semibold text-slate-500 dark:text-slate-400">Method:</span>
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <input
                          type="radio"
                          name="wordMethod"
                          checked={effectiveMethod === "occlusion"}
                          onChange={() => setWordMethod("occlusion")}
                          className="accent-blue-600"
                        />
                        <span>Occlusion (remove the word, see how the score changes)</span>
                      </label>
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <input
                          type="radio"
                          name="wordMethod"
                          checked={effectiveMethod === "grad_x_input"}
                          disabled={explanation.model === "baseline"}
                          onChange={() => setWordMethod("grad_x_input")}
                          className="accent-blue-600"
                        />
                        <span className={explanation.model === "baseline" ? "opacity-50" : ""}>
                          Gradient × input (backpropagation)
                        </span>
                      </label>
                    </div>

                    {/* Highlighted HTML Words */}
                    <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 leading-relaxed">
                      {explanation.tokens.map((t) => {
                        const v = Number(t[effectiveMethod] || 0);
                        const a = Math.min(Math.abs(v) / maxTokenAbs, 1.0) * 0.85;
                        const bg =
                          v > 0
                            ? `rgba(41, 128, 185, ${a.toFixed(2)})`
                            : `rgba(192, 57, 43, ${a.toFixed(2)})`;
                        return (
                          <span
                            key={t.position}
                            title={`${v >= 0 ? "+" : ""}${v.toFixed(3)}`}
                            style={{
                              background: bg,
                              padding: "1px 4px",
                              margin: "1.5px",
                              borderRadius: "3px",
                              display: "inline-block",
                            }}
                            className="text-sm font-mono cursor-help"
                          >
                            {t.token}
                          </span>
                        );
                      })}
                    </div>

                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      🔵 pushes toward <strong>Real</strong> · 🔴 pushes toward <strong>Fake</strong> · darker = stronger. Hover a word for its value. Words not in the vocabulary were replaced by <code>&lt;unk&gt;</code> and carry no information.
                    </p>

                    {/* Top 12 Influential Words Table */}
                    <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead className="bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-300">
                          <tr>
                            <th className="py-2 px-3">position</th>
                            <th className="py-2 px-3">token</th>
                            <th className="py-2 px-3">in_vocab</th>
                            <th className="py-2 px-3">{effectiveMethod}</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
                          {top12Words.map((w) => {
                            const val = Number(w[effectiveMethod] || 0);
                            return (
                              <tr key={w.position}>
                                <td className="py-2 px-3">{w.position}</td>
                                <td className="py-2 px-3 font-bold">{w.token}</td>
                                <td className="py-2 px-3">{w.in_vocab ? "true" : "false"}</td>
                                <td className={`py-2 px-3 font-bold ${val >= 0 ? "text-blue-500" : "text-rose-500"}`}>
                                  {val >= 0 ? `+${val.toFixed(4)}` : val.toFixed(4)}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* TAB 4: THE FINAL SUM */}
                {activeExplainTab === "sum" && (
                  <div className="space-y-4">
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      The exact arithmetic that produced the answer.
                    </p>
                    <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
                      <table className="w-full text-left text-xs sm:text-sm border-collapse">
                        <thead className="bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-300">
                          <tr>
                            <th className="py-2.5 px-4">part</th>
                            <th className="py-2.5 px-4">value</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
                          <tr>
                            <td className="py-2.5 px-4 font-sans">Sum of contributions pushing toward REAL</td>
                            <td className="py-2.5 px-4 text-blue-500 font-bold">
                              +{explanation.summary.sum_push_real.toFixed(4)}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2.5 px-4 font-sans">Sum of contributions pushing toward FAKE</td>
                            <td className="py-2.5 px-4 text-rose-500 font-bold">
                              {explanation.summary.sum_push_fake.toFixed(4)}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2.5 px-4 font-sans">Bias (the model's starting lean)</td>
                            <td className="py-2.5 px-4">
                              {explanation.summary.bias >= 0
                                ? `+${explanation.summary.bias.toFixed(4)}`
                                : explanation.summary.bias.toFixed(4)}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2.5 px-4 font-sans font-bold">z = total (logit)</td>
                            <td className="py-2.5 px-4 font-bold">
                              {explanation.logit_from_parts >= 0
                                ? `+${explanation.logit_from_parts.toFixed(4)}`
                                : explanation.logit_from_parts.toFixed(4)}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2.5 px-4 font-sans font-bold">P(Real) = 1 / (1 + e^-z)</td>
                            <td className="py-2.5 px-4 font-bold text-emerald-500">
                              {explanation.p_real.toFixed(4)}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2.5 px-4 font-sans">Decision rule</td>
                            <td className="py-2.5 px-4 font-sans">
                              P(Real) ≥ 0.5 → Real, else Fake ⇒ <strong>{explanation.verdict}</strong>
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                      Check: the parts add up to the model's own output (difference{" "}
                      {explanation.logit_diff.toExponential(1)}).
                    </p>
                  </div>
                )}
              </section>
            )}
          </>
        )}

        {/* =====================================================================
            MAIN TAB 2: MODEL COMPARISON (EXACT PARITY WITH FRIEND'S APP.PY)
           ===================================================================== */}
        {mainTab === "comparison" && (
          <section className="space-y-6">
            <div className="bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
              <div>
                <h2 className="text-base font-bold text-slate-900 dark:text-white">
                  Held-Out Test Split Model Comparison
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  All results below come from the held-out test split. Higher scores are not evidence of fact verification or guaranteed performance on new sources.
                </p>
              </div>

              <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
                <table className="w-full text-left text-xs border-collapse">
                  <thead className="bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-300">
                    <tr>
                      <th className="py-2.5 px-3">Model</th>
                      <th className="py-2.5 px-3">Accuracy</th>
                      <th className="py-2.5 px-3">Macro F1</th>
                      <th className="py-2.5 px-3">ROC AUC</th>
                      <th className="py-2.5 px-3">Fake precision</th>
                      <th className="py-2.5 px-3">Fake recall</th>
                      <th className="py-2.5 px-3">Fake F1</th>
                      <th className="py-2.5 px-3">Fake support</th>
                      <th className="py-2.5 px-3">Real precision</th>
                      <th className="py-2.5 px-3">Real recall</th>
                      <th className="py-2.5 px-3">Real F1</th>
                      <th className="py-2.5 px-3">Real support</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
                    {Object.entries(metrics).map(([name, m]) => {
                      const fake = m.per_class?.Fake || {};
                      const real = m.per_class?.Real || {};
                      return (
                        <tr key={name}>
                          <td className="py-2.5 px-3 font-sans font-bold">{MODEL_META[name]?.nice || name}</td>
                          <td className="py-2.5 px-3">{m.accuracy?.toFixed(4)}</td>
                          <td className="py-2.5 px-3 font-bold text-blue-500">{m.macro_f1?.toFixed(4)}</td>
                          <td className="py-2.5 px-3">{m.roc_auc?.toFixed(4)}</td>
                          <td className="py-2.5 px-3">{fake.precision?.toFixed(4)}</td>
                          <td className="py-2.5 px-3">{fake.recall?.toFixed(4)}</td>
                          <td className="py-2.5 px-3">{fake.f1?.toFixed(4)}</td>
                          <td className="py-2.5 px-3">{fake.support}</td>
                          <td className="py-2.5 px-3">{real.precision?.toFixed(4)}</td>
                          <td className="py-2.5 px-3">{real.recall?.toFixed(4)}</td>
                          <td className="py-2.5 px-3">{real.f1?.toFixed(4)}</td>
                          <td className="py-2.5 px-3">{real.support}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Test Metric Comparison Bar Chart & Confusion Matrix Viewer */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Bar Chart */}
              <div className="bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold">Test metric comparison</h3>
                  <select
                    value={chartMetric}
                    onChange={(e) => setChartMetric(e.target.value)}
                    className="bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold"
                  >
                    <option value="Accuracy">Accuracy</option>
                    <option value="Macro F1">Macro F1</option>
                    <option value="ROC AUC">ROC AUC</option>
                  </select>
                </div>

                <div className="space-y-3 pt-2">
                  {Object.entries(metrics).map(([name, m]) => {
                    const rawVal =
                      chartMetric === "Accuracy"
                        ? m.accuracy
                        : chartMetric === "Macro F1"
                        ? m.macro_f1
                        : m.roc_auc;
                    const pct = (rawVal || 0) * 100;
                    return (
                      <div key={name} className="space-y-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="font-sans font-semibold">{MODEL_META[name]?.nice || name}</span>
                          <span className="font-bold text-blue-500">{rawVal?.toFixed(4)}</span>
                        </div>
                        <div className="w-full h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 rounded-full transition-all duration-500"
                            style={{ width: `${Math.max(5, pct)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Confusion Matrix Table */}
              <div className="bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold">Confusion matrix</h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      Rows are actual labels; columns are predicted labels.
                    </p>
                  </div>
                  <select
                    value={confusionModel}
                    onChange={(e) => setConfusionModel(e.target.value)}
                    className="bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold"
                  >
                    {Object.keys(metrics).map((m) => (
                      <option key={m} value={m}>
                        {MODEL_META[m]?.nice || m}
                      </option>
                    ))}
                  </select>
                </div>

                {metrics[confusionModel]?.confusion_matrix && (
                  <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl">
                    <table className="w-full text-left text-xs sm:text-sm border-collapse font-mono">
                      <thead className="bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-300">
                        <tr>
                          <th className="py-3 px-4"></th>
                          <th className="py-3 px-4">Predicted Fake</th>
                          <th className="py-3 px-4">Predicted Real</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                        <tr>
                          <td className="py-3 px-4 font-sans font-bold">Actual Fake</td>
                          <td className="py-3 px-4 font-bold text-emerald-500">
                            {metrics[confusionModel].confusion_matrix[0][0]}
                          </td>
                          <td className="py-3 px-4 text-rose-500">
                            {metrics[confusionModel].confusion_matrix[0][1]}
                          </td>
                        </tr>
                        <tr>
                          <td className="py-3 px-4 font-sans font-bold">Actual Real</td>
                          <td className="py-3 px-4 text-rose-500">
                            {metrics[confusionModel].confusion_matrix[1][0]}
                          </td>
                          <td className="py-3 px-4 font-bold text-emerald-500">
                            {metrics[confusionModel].confusion_matrix[1][1]}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}