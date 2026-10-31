import { useState, useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000";

const MODEL_META = {
  lstm: {
    title: "BiLSTM",
    subtitle: "Bidirectional LSTM",
    badge: "Recurrent • 384-d Pool",
    desc: "Captures long-range forward & backward sequential context.",
  },
  cnn: {
    title: "TextCNN",
    subtitle: "1D ConvNet (k=3,4,5)",
    badge: "Convolutional • N-Grams",
    desc: "Detects local phrase patterns & sensational n-gram triggers.",
  },
  gru: {
    title: "BiGRU",
    subtitle: "Gated Recurrent Unit",
    badge: "Fast Recurrent • 2-Gate",
    desc: "Streamlined recurrent gating with lower parameter overhead.",
  },
  baseline: {
    title: "TF-IDF + LR",
    subtitle: "Logistic Regression",
    badge: "Classical ML • 50k Features",
    desc: "Unigram & bigram frequency baseline tuned on validation F1.",
  },
};

const SAMPLE_ARTICLES = [
  {
    label: "Load Real News Sample",
    type: "real",
    text: "The central bank announced on Thursday that benchmark interest rates would remain unchanged following a two-day monetary policy review. Officials cited moderating inflation across food and energy sectors while projecting steady gross domestic product growth over the upcoming fiscal quarter. Economists surveyed ahead of the decision largely anticipated the hold as policymakers balance price stability with industrial credit demand.",
  },
  {
    label: "Load Fake News Sample",
    type: "fake",
    text: "SHOCKING: Secret leaked documents prove that global weather satellites are being used to mind-control voters using invisible frequency beams! Mainstream media refuses to air this explosive truth after whistleblowers revealed that every smartphone battery is secretly recording your thoughts and sending them to an underground bunker. Share this immediately before they delete the internet!",
  },
];

export default function App() {
  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState("lstm");
  const [installed, setInstalled] = useState(["lstm", "cnn", "gru", "baseline"]);
  const [metrics, setMetrics] = useState({});
  const [singleResult, setSingleResult] = useState(null);
  const [allResults, setAllResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");

  // Theme management
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

  // Fetch models & metrics from FastAPI
  useEffect(() => {
    fetch(`${API_BASE}/api/models`)
      .then((r) => r.json())
      .then((data) => {
        setApiOnline(true);
        if (data.installed?.length) setInstalled(data.installed);
        if (data.metrics) setMetrics(data.metrics);
      })
      .catch(() => setApiOnline(false));
  }, []);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

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
      setApiOnline(true);
      if (data.error) setError(data.error);
      else if (compareAll) setAllResults(data.results || []);
      else setSingleResult(data);
    } catch {
      setApiOnline(false);
      setError("FastAPI backend is offline. Start it with: uvicorn src.api:app --port 8000");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-slate-50 dark:bg-[#090D16] text-slate-900 dark:text-slate-100 transition-colors duration-300 relative overflow-x-hidden">
      {/* Subtle Ambient Gradient Glow */}
      <div className="pointer-events-none fixed inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
        <div
          className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-blue-600 to-indigo-500 opacity-15 dark:opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"
        />
      </div>

      {/* Top Sticky Glassmorphic Navbar */}
      <nav className="sticky top-0 z-30 border-b border-slate-200/80 dark:border-slate-800/80 bg-white/75 dark:bg-[#090D16]/75 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
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
                Leakage-Controlled Deep Learning & NLP Verification Suite
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Live API Status Pill */}
            <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border border-slate-200 dark:border-slate-800 bg-slate-100/70 dark:bg-slate-900/70">
              <span className={`h-2 w-2 rounded-full ${apiOnline ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
              <span className="text-slate-600 dark:text-slate-300">
                {apiOnline ? "Inference Engine Ready" : "API Offline"}
              </span>
            </div>

            {/* Clean Segmented Theme Toggler */}
            <div className="inline-flex items-center rounded-xl p-1 bg-slate-200/70 dark:bg-slate-900 border border-slate-300/60 dark:border-slate-800">
              {[
                { id: "light", label: "Light", icon: "M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" },
                { id: "dark", label: "Dark", icon: "M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z" },
                { id: "system", label: "Auto", icon: "M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115 18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0V12a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 12V5.25" },
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => setTheme(item.id)}
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    theme === item.id
                      ? "bg-white dark:bg-slate-800 text-blue-600 dark:text-blue-400 shadow-sm"
                      : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                  }`}
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={item.icon} />
                  </svg>
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Step 1: Interactive Neural Architecture Selector Cards */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              1. Select Classification Architecture
            </h2>
            <span className="text-xs text-slate-400 dark:text-slate-500">
              Trained on 70% Stratified ISOT Split • Publisher Shortcuts Stripped
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
            {["lstm", "cnn", "gru", "baseline"].map((key) => {
              const info = MODEL_META[key];
              const isSelected = selectedModel === key;
              const isAvailable = installed.includes(key);
              const modelMetric = metrics[key];

              return (
                <button
                  key={key}
                  onClick={() => setSelectedModel(key)}
                  className={`text-left p-4 rounded-2xl border transition-all duration-200 relative overflow-hidden ${
                    isSelected
                      ? "bg-blue-50/70 dark:bg-blue-950/30 border-blue-500 dark:border-blue-500 ring-2 ring-blue-500/20 shadow-md"
                      : "bg-white dark:bg-slate-900/80 border-slate-200/90 dark:border-slate-800/90 hover:border-slate-300 dark:hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-base font-bold text-slate-900 dark:text-white">
                      {info.title}
                    </span>
                    {modelMetric ? (
                      <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded-md bg-emerald-100 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300 border border-emerald-300/50 dark:border-emerald-700/50">
                        F1: {(modelMetric.macro_f1 * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-[11px] px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-500">
                        {isAvailable ? "Ready" : "Untrained"}
                      </span>
                    )}
                  </div>
                  <div className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-1.5">
                    {info.badge}
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                    {info.desc}
                  </p>
                </button>
              );
            })}
          </div>
        </section>

        {/* Step 2: Main Split Workspace (Editor on Left, Live Telemetry on Right) */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Article Editor (7 cols) */}
          <div className="lg:col-span-7 bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 rounded-2xl shadow-sm overflow-hidden">
            {/* Editor Toolbar */}
            <div className="px-5 py-3.5 border-b border-slate-200/80 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2 bg-slate-50/60 dark:bg-slate-900/50">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                2. Article Input Stream
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
                    }}
                    className="text-xs font-medium px-2.5 py-1 rounded-lg text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

            {/* Textarea */}
            <div className="p-5">
              <textarea
                rows={8}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste a news headline and full article body here, or click a sample button above to test..."
                className="w-full bg-slate-50/70 dark:bg-[#0B0F19] border border-slate-200 dark:border-slate-800 rounded-xl p-4 text-sm leading-relaxed text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 transition resize-y"
              />

              {/* Footer Bar with Stats & Action Buttons */}
              <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-4 text-xs font-mono text-slate-500 dark:text-slate-400">
                  <span>{wordCount} words</span>
                  <span>•</span>
                  <span>{text.length.toLocaleString()} / 20,000 chars</span>
                </div>

                <div className="flex flex-wrap items-center gap-2.5">
                  <button
                    onClick={() => handleAnalyze(true)}
                    disabled={loading}
                    className="px-4 py-2.5 rounded-xl text-xs font-semibold border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 transition disabled:opacity-50"
                  >
                    Compare All 4 Models
                  </button>
                  <button
                    onClick={() => handleAnalyze(false)}
                    disabled={loading}
                    className="px-5 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-md shadow-blue-500/20 transition disabled:opacity-50"
                  >
                    {loading ? "Running Inference..." : `Analyze with ${MODEL_META[selectedModel]?.title}`}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Live Diagnostic Output Panel (5 cols) */}
          <div className="lg:col-span-5 space-y-4">
            <div className="bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 rounded-2xl p-5 shadow-sm min-h-[310px] flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between border-b border-slate-200/80 dark:border-slate-800 pb-3 mb-4">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                    3. Neural Diagnostic Report
                  </span>
                  <span className="text-xs font-mono text-slate-400">
                    Threshold: P(Real) ≥ 0.50
                  </span>
                </div>

                {error && (
                  <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/50 border border-rose-300 dark:border-rose-800 text-rose-700 dark:text-rose-200 text-xs leading-relaxed">
                    <strong>Inference Error:</strong> {error}
                  </div>
                )}

                {!error && !singleResult && allResults.length === 0 && (
                  <div className="py-12 text-center space-y-2">
                    <div className="mx-auto w-12 h-12 rounded-2xl bg-slate-100 dark:bg-slate-800/80 flex items-center justify-center text-slate-400">
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.7}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
                      </svg>
                    </div>
                    <p className="text-sm font-medium text-slate-600 dark:text-slate-300">
                      Awaiting Article Input
                    </p>
                    <p className="text-xs text-slate-400 max-w-xs mx-auto">
                      Select an architecture above and run analysis to view confidence scores and sub-100ms latency metrics.
                    </p>
                  </div>
                )}

                {/* Single Model Prediction Card */}
                {singleResult && (
                  <div className="space-y-4">
                    <div
                      className={`p-4 rounded-xl border ${
                        singleResult.label === "Real"
                          ? "bg-emerald-50/70 dark:bg-emerald-950/30 border-emerald-400/70 dark:border-emerald-700"
                          : "bg-rose-50/70 dark:bg-rose-950/30 border-rose-400/70 dark:border-rose-700"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                          {singleResult.model.toUpperCase()} Verdict
                        </span>
                        <span className="text-xs font-mono px-2 py-0.5 rounded bg-white/70 dark:bg-slate-900/70">
                          {singleResult.latency_ms} ms
                        </span>
                      </div>
                      <div className="text-xl font-extrabold mt-1">
                        {singleResult.label === "Real"
                          ? "Likely Authentic (Real-like)"
                          : "Potential Misinformation (Fake-like)"}
                      </div>
                    </div>

                    {/* Probability Progress Bar */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs font-medium">
                        <span className="text-rose-500">0% (Fake)</span>
                        <span className="font-mono font-bold text-slate-800 dark:text-slate-200">
                          P(Real) = {(singleResult.score_real * 100).toFixed(2)}%
                        </span>
                        <span className="text-emerald-500">100% (Real)</span>
                      </div>
                      <div className="w-full h-3 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 rounded-full ${
                            singleResult.label === "Real"
                              ? "bg-gradient-to-r from-emerald-500 to-teal-400"
                              : "bg-gradient-to-r from-rose-600 to-orange-500"
                          }`}
                          style={{ width: `${Math.max(4, singleResult.score_real * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Side-by-Side Comparison List */}
                {allResults.length > 0 && (
                  <div className="space-y-2.5">
                    {allResults.map((r) => (
                      <div
                        key={r.model}
                        className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-950/50 flex items-center justify-between"
                      >
                        <div>
                          <div className="text-xs font-bold text-slate-800 dark:text-slate-200">
                            {r.model}
                          </div>
                          <div className="text-[11px] font-mono text-slate-400">
                            Latency: {r.latency_ms} ms
                          </div>
                        </div>
                        <div className="text-right">
                          <span
                            className={`inline-block text-xs font-bold px-2.5 py-1 rounded-lg ${
                              r.label === "Real"
                                ? "bg-emerald-100 dark:bg-emerald-950/70 text-emerald-700 dark:text-emerald-300 border border-emerald-400/40"
                                : "bg-rose-100 dark:bg-rose-950/70 text-rose-700 dark:text-rose-300 border border-rose-400/40"
                            }`}
                          >
                            {r.label === "Real" ? "REAL" : "FAKE"} • {r.confidence}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Subtle Footer Note inside Diagnostic Card */}
              <div className="mt-5 pt-3 border-t border-slate-200/70 dark:border-slate-800/80 text-[11px] text-slate-400 dark:text-slate-500 leading-normal">
                <strong>Methodology Note:</strong> Predictions reflect linguistic patterns learned from the ISOT dataset and do not verify factual claims independently.
              </div>
            </div>
          </div>
        </section>

        {/* Step 3: Held-Out 15% Test Set Benchmark Table */}
        {Object.keys(metrics).length > 0 && (
          <section className="bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 rounded-2xl p-6 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
              <div>
                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-700 dark:text-slate-200">
                  Held-Out 15% Test Split Benchmark (5,820 Articles)
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Evaluated once on identical deduplicated, tag-stripped test articles.
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs sm:text-sm border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 uppercase text-[11px] tracking-wider">
                    <th className="py-3 px-3">Architecture</th>
                    <th className="py-3 px-3">Accuracy</th>
                    <th className="py-3 px-3">Macro-F1</th>
                    <th className="py-3 px-3">ROC-AUC</th>
                    <th className="py-3 px-3">Fake Recall</th>
                    <th className="py-3 px-3">Real Recall</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70 font-mono">
                  {Object.entries(metrics).map(([name, m]) => (
                    <tr key={name} className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition">
                      <td className="py-3 px-3 font-sans font-bold uppercase text-slate-900 dark:text-white">
                        {MODEL_META[name]?.title || name}
                      </td>
                      <td className="py-3 px-3">{(m.accuracy * 100).toFixed(2)}%</td>
                      <td className="py-3 px-3 font-bold text-blue-600 dark:text-blue-400">
                        {(m.macro_f1 * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3">{m.roc_auc ? m.roc_auc.toFixed(4) : "—"}</td>
                      <td className="py-3 px-3">{(m.per_class.Fake.recall * 100).toFixed(2)}%</td>
                      <td className="py-3 px-3">{(m.per_class.Real.recall * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}