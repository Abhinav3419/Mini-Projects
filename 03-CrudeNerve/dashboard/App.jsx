import { useState, useEffect, useCallback } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const API = "http://localhost:8000";

const MODES = [
  { id: "intuitive", label: "Intuitive", desc: "All signals at face value", icon: "→" },
  { id: "counter_intuitive", label: "Counter-intuitive", desc: "Signals are misleading", icon: "↻" },
  { id: "golden_mean", label: "Golden mean", desc: "Selective trust per signal", icon: "◎" },
];

const CI_SUBS = [
  { id: "flip", label: "Flip direction" },
  { id: "zero", label: "Zero out" },
];

const PRESETS = {
  macro_only: { label: "Macro only", desc: "Ignore social noise" },
  social_skeptic: { label: "Social skeptic", desc: "Discount Trump rhetoric" },
  crisis_trader: { label: "Crisis trader", desc: "Real-time signals only" },
  contrarian_lite: { label: "Contrarian lite", desc: "Hard supply data only" },
};

const STREAMS = [
  { id: "truth_social", label: "Truth Social", cat: "social" },
  { id: "twitter", label: "Twitter/X", cat: "social" },
  { id: "gdelt_tension", label: "GDELT tension", cat: "geopolitical" },
  { id: "dho_decay", label: "DHO decay", cat: "geopolitical" },
  { id: "eia_inventory", label: "EIA inventory", cat: "supply" },
  { id: "spr_releases", label: "SPR releases", cat: "supply" },
  { id: "hormuz_shipping", label: "Hormuz shipping", cat: "supply" },
  { id: "opec_compliance", label: "OPEC compliance", cat: "supply" },
];

const WAR_LEVELS = [
  "Diplomatic calm", "Background tension", "Diplomatic friction",
  "Sanctions escalation", "Proxy conflict", "Direct threats",
  "Military mobilization", "Limited strikes", "Open conflict", "Full-scale war"
];

const COLORS = {
  bg: "#0B0E11", card: "#12161C", cardHover: "#181D25",
  border: "#1E2530", borderActive: "#2A3544",
  text: "#E8ECF1", textMuted: "#7B8794", textDim: "#4A5568",
  accent: "#E85D24", accentDim: "#C04828",
  bullish: "#22C55E", bearish: "#EF4444", neutral: "#6B7280",
  teal: "#14B8A6", amber: "#F59E0B", purple: "#8B5CF6",
  war1: "#22C55E", war5: "#F59E0B", war10: "#EF4444",
};

function getWarColor(level) {
  if (level <= 3) return COLORS.bullish;
  if (level <= 6) return COLORS.amber;
  return COLORS.bearish;
}

function pct(v) { return `${(v * 100).toFixed(1)}%`; }

export default function App() {
  const [mode, setMode] = useState("intuitive");
  const [ciSub, setCiSub] = useState("flip");
  const [warLevel, setWarLevel] = useState(null);
  const [trustConfig, setTrustConfig] = useState({
    truth_social: false, twitter: true, gdelt_tension: true,
    dho_decay: true, eia_inventory: true, spr_releases: true,
    hormuz_shipping: true, opec_compliance: true,
  });
  const [preset, setPreset] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runPrediction = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const body = {
        mode,
        ci_submode: ciSub,
        war_level: warLevel,
        trust_config: mode === "golden_mean" ? trustConfig : undefined,
        preset: mode === "golden_mean" ? preset : undefined,
      };
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
      setResult(mockResult());
    } finally {
      setLoading(false);
    }
  }, [mode, ciSub, warLevel, trustConfig, preset]);

  useEffect(() => { setResult(mockResult()); }, []);

  const wa = result?.war_assessment;
  const preds = result?.predictions || [];
  const drivers = result?.top_drivers || [];
  const displayLevel = warLevel || wa?.recommended_level || 4;

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text, fontFamily: "'JetBrains Mono', 'SF Mono', monospace", padding: "0" }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ borderBottom: `1px solid ${COLORS.border}`, padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: COLORS.accent, boxShadow: `0 0 12px ${COLORS.accent}60` }} />
          <span style={{ fontSize: 15, fontWeight: 500, letterSpacing: 2 }}>CRUDENERVE</span>
          <span style={{ fontSize: 11, color: COLORS.textMuted, letterSpacing: 1 }}>v3.1</span>
        </div>
        <button onClick={runPrediction} disabled={loading}
          style={{ background: COLORS.accent, color: "#fff", border: "none", padding: "8px 20px", borderRadius: 6, cursor: "pointer", fontSize: 12, fontWeight: 500, letterSpacing: 1, fontFamily: "inherit", opacity: loading ? 0.6 : 1 }}>
          {loading ? "RUNNING..." : "RUN PREDICTION"}
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr 300px", gap: 0, minHeight: "calc(100vh - 53px)" }}>

        {/* LEFT PANEL — Controls */}
        <div style={{ borderRight: `1px solid ${COLORS.border}`, padding: 16, overflowY: "auto" }}>

          {/* Mode selector */}
          <SectionLabel>Prediction mode</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 16 }}>
            {MODES.map(m => (
              <button key={m.id} onClick={() => { setMode(m.id); setPreset(null); }}
                style={{ background: mode === m.id ? COLORS.cardHover : "transparent", border: `1px solid ${mode === m.id ? COLORS.borderActive : COLORS.border}`, borderRadius: 6, padding: "8px 10px", cursor: "pointer", textAlign: "left", color: COLORS.text, fontFamily: "inherit", fontSize: 11 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 14, color: mode === m.id ? COLORS.accent : COLORS.textDim }}>{m.icon}</span>
                  <span style={{ fontWeight: 500 }}>{m.label}</span>
                </div>
                <div style={{ fontSize: 10, color: COLORS.textMuted, marginTop: 2, paddingLeft: 20 }}>{m.desc}</div>
              </button>
            ))}
          </div>

          {/* CI sub-mode */}
          {mode === "counter_intuitive" && (
            <>
              <SectionLabel>Contrarian sub-mode</SectionLabel>
              <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
                {CI_SUBS.map(s => (
                  <button key={s.id} onClick={() => setCiSub(s.id)}
                    style={{ flex: 1, background: ciSub === s.id ? COLORS.cardHover : "transparent", border: `1px solid ${ciSub === s.id ? COLORS.borderActive : COLORS.border}`, borderRadius: 6, padding: "6px 8px", cursor: "pointer", color: COLORS.text, fontFamily: "inherit", fontSize: 10, fontWeight: ciSub === s.id ? 500 : 400 }}>
                    {s.label}
                  </button>
                ))}
              </div>
            </>
          )}

          {/* Golden Mean controls */}
          {mode === "golden_mean" && (
            <>
              <SectionLabel>Presets</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, marginBottom: 12 }}>
                {Object.entries(PRESETS).map(([k, v]) => (
                  <button key={k} onClick={() => setPreset(preset === k ? null : k)}
                    style={{ background: preset === k ? COLORS.cardHover : "transparent", border: `1px solid ${preset === k ? COLORS.accent + "40" : COLORS.border}`, borderRadius: 6, padding: "6px 8px", cursor: "pointer", color: COLORS.text, fontFamily: "inherit", fontSize: 10, textAlign: "left" }}>
                    <div style={{ fontWeight: 500 }}>{v.label}</div>
                    <div style={{ fontSize: 9, color: COLORS.textMuted }}>{v.desc}</div>
                  </button>
                ))}
              </div>

              <SectionLabel>Signal streams</SectionLabel>
              <div style={{ display: "flex", flexDirection: "column", gap: 3, marginBottom: 16 }}>
                {STREAMS.map(s => (
                  <label key={s.id} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", padding: "4px 6px", borderRadius: 4, fontSize: 11 }}>
                    <input type="checkbox" checked={trustConfig[s.id]}
                      onChange={e => setTrustConfig(prev => ({ ...prev, [s.id]: e.target.checked }))}
                      style={{ accentColor: COLORS.accent }} />
                    <span>{s.label}</span>
                    <span style={{ fontSize: 9, color: COLORS.textDim, marginLeft: "auto" }}>{s.cat}</span>
                  </label>
                ))}
              </div>
            </>
          )}

          {/* War barometer */}
          <SectionLabel>War tension level</SectionLabel>
          <div style={{ marginBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: COLORS.textMuted, marginBottom: 4 }}>
              <span>1 Calm</span>
              <span>10 War</span>
            </div>
            <input type="range" min={1} max={10} value={displayLevel}
              onChange={e => setWarLevel(parseInt(e.target.value))}
              style={{ width: "100%", accentColor: getWarColor(displayLevel) }} />
            <div style={{ textAlign: "center", marginTop: 6 }}>
              <span style={{ fontSize: 28, fontWeight: 600, color: getWarColor(displayLevel) }}>{displayLevel}</span>
              <div style={{ fontSize: 10, color: COLORS.textMuted }}>{WAR_LEVELS[displayLevel - 1]}</div>
              {wa && <div style={{ fontSize: 9, color: COLORS.textDim, marginTop: 2 }}>recommended: {wa.recommended_level} ({pct(wa.confidence)} confidence)</div>}
            </div>
            <button onClick={() => setWarLevel(null)} style={{ width: "100%", marginTop: 6, background: "transparent", border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "4px", cursor: "pointer", color: COLORS.textMuted, fontFamily: "inherit", fontSize: 9 }}>
              Use auto-recommended
            </button>
          </div>
        </div>

        {/* CENTER — Predictions */}
        <div style={{ padding: 20, overflowY: "auto" }}>

          {error && <div style={{ background: "#EF444420", border: "1px solid #EF444440", borderRadius: 6, padding: "8px 12px", fontSize: 11, color: COLORS.bearish, marginBottom: 12 }}>Using demo data — connect API at localhost:8000</div>}

          {/* Horizon predictions */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 20 }}>
            {preds.map(p => (
              <div key={p.horizon_hours} style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 16 }}>
                <div style={{ fontSize: 10, color: COLORS.textMuted, marginBottom: 8, letterSpacing: 1 }}>{p.horizon_hours}H FORECAST</div>
                <div style={{ fontSize: 32, fontWeight: 600, color: p.vix_direction_prob > 0.6 ? COLORS.bearish : p.vix_direction_prob > 0.4 ? COLORS.amber : COLORS.bullish }}>
                  {pct(p.vix_direction_prob)}
                </div>
                <div style={{ fontSize: 10, color: COLORS.textMuted, marginTop: 4 }}>VIX move probability</div>
                <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                  <MiniBar label="Up" value={p.vix_up_prob} color={COLORS.bearish} />
                  <MiniBar label="Down" value={p.vix_down_prob} color={COLORS.bullish} />
                </div>
                {p.brent_return_quantiles && (
                  <div style={{ marginTop: 10, fontSize: 9, color: COLORS.textDim }}>
                    <div>Brent return range</div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2 }}>
                      <span style={{ color: COLORS.bullish }}>{(p.brent_return_quantiles.q10 * 100).toFixed(1)}%</span>
                      <span style={{ color: COLORS.text, fontWeight: 500 }}>{(p.brent_return_quantiles.q50 * 100).toFixed(1)}%</span>
                      <span style={{ color: COLORS.bearish }}>+{(p.brent_return_quantiles.q90 * 100).toFixed(1)}%</span>
                    </div>
                    <QuantileFan quantiles={p.brent_return_quantiles} />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Mode info */}
          {result?.mode_info && (
            <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: 14, marginBottom: 12 }}>
              <div style={{ fontSize: 10, color: COLORS.textMuted, letterSpacing: 1, marginBottom: 6 }}>ACTIVE MODE</div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>{result.mode_info.mode}</div>
              <div style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 2 }}>{result.mode_info.philosophy}</div>
              {result.mode_info.risk && <div style={{ fontSize: 10, color: COLORS.bearish, marginTop: 6, opacity: 0.8 }}>Risk: {result.mode_info.risk}</div>}
            </div>
          )}

          {/* Processing info */}
          {result && (
            <div style={{ fontSize: 9, color: COLORS.textDim, display: "flex", gap: 16 }}>
              <span>{result.feature_count} features</span>
              <span>{result.processing_time_ms?.toFixed(0)}ms</span>
              <span>{result.data_date_range?.start} → {result.data_date_range?.end}</span>
            </div>
          )}
        </div>

        {/* RIGHT PANEL — Drivers + War assessment */}
        <div style={{ borderLeft: `1px solid ${COLORS.border}`, padding: 16, overflowY: "auto" }}>

          {/* SHAP drivers */}
          <SectionLabel>Top drivers</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 3, marginBottom: 20 }}>
            {drivers.slice(0, 8).map((d, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 8px", background: i < 3 ? COLORS.card : "transparent", borderRadius: 4, fontSize: 10 }}>
                <div style={{ width: 4, height: 4, borderRadius: "50%", background: d.direction === "bearish" ? COLORS.bearish : COLORS.bullish }} />
                <span style={{ flex: 1, color: i < 3 ? COLORS.text : COLORS.textMuted }}>{d.feature}</span>
                <span style={{ fontWeight: 500, color: d.shap_value > 0 ? COLORS.bearish : COLORS.bullish }}>{d.shap_value > 0 ? "+" : ""}{d.shap_value.toFixed(3)}</span>
              </div>
            ))}
          </div>

          {/* War sub-scores */}
          {wa?.sub_scores && (
            <>
              <SectionLabel>War assessment breakdown</SectionLabel>
              <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 16 }}>
                {Object.entries(wa.sub_scores).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 10 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                      <span style={{ color: COLORS.textMuted }}>{k.replace(/_/g, " ")}</span>
                      <span style={{ color: getWarColor(v), fontWeight: 500 }}>{v.toFixed(1)}</span>
                    </div>
                    <div style={{ height: 3, background: COLORS.border, borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${v * 10}%`, background: getWarColor(v), borderRadius: 2, transition: "width 0.3s" }} />
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 9, color: COLORS.textDim }}>
                <div>Brent impact: {wa.brent_impact_range}</div>
                <div>VIX impact: {wa.vix_impact_range}</div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function SectionLabel({ children }) {
  return <div style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 8, fontWeight: 500 }}>{children}</div>;
}

function MiniBar({ label, value, color }) {
  return (
    <div style={{ flex: 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: COLORS.textMuted, marginBottom: 2 }}>
        <span>{label}</span><span style={{ color }}>{pct(value)}</span>
      </div>
      <div style={{ height: 3, background: COLORS.border, borderRadius: 2, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${value * 100}%`, background: color, borderRadius: 2 }} />
      </div>
    </div>
  );
}

function QuantileFan({ quantiles }) {
  if (!quantiles) return null;
  const vals = [quantiles.q10, quantiles.q25, quantiles.q50, quantiles.q75, quantiles.q90].map(v => v * 100);
  const min = Math.min(...vals) - 1;
  const max = Math.max(...vals) + 1;
  const range = max - min || 1;
  const pos = v => ((v - min) / range) * 100;

  return (
    <div style={{ position: "relative", height: 14, marginTop: 4, background: COLORS.border, borderRadius: 2 }}>
      <div style={{ position: "absolute", left: `${pos(vals[0])}%`, right: `${100 - pos(vals[4])}%`, top: 3, height: 8, background: `${COLORS.teal}25`, borderRadius: 2 }} />
      <div style={{ position: "absolute", left: `${pos(vals[1])}%`, right: `${100 - pos(vals[3])}%`, top: 1, height: 12, background: `${COLORS.teal}40`, borderRadius: 2 }} />
      <div style={{ position: "absolute", left: `${pos(vals[2])}%`, top: 0, width: 2, height: 14, background: COLORS.teal, borderRadius: 1 }} />
    </div>
  );
}

function mockResult() {
  return {
    mode_info: { mode: "Intuitive", philosophy: "All signals at face value", risk: "Gets killed by reversals and bluffs" },
    war_assessment: {
      recommended_level: 6, level_name: "Direct threats", confidence: 0.35,
      sub_scores: { gdelt_military: 4.0, truth_social_war: 5.0, twitter_fear: 3.0, hormuz_status: 7.0, oil_volatility: 2.0 },
      brent_impact_range: "+5% to +12%", vix_impact_range: "+3% to +6%",
    },
    predictions: [
      { horizon_hours: 24, vix_direction_prob: 0.58, vix_up_prob: 0.38, vix_down_prob: 0.20, brent_return_quantiles: { q10: -0.02, q25: 0.005, q50: 0.018, q75: 0.035, q90: 0.06 } },
      { horizon_hours: 48, vix_direction_prob: 0.64, vix_up_prob: 0.42, vix_down_prob: 0.22, brent_return_quantiles: { q10: -0.015, q25: 0.01, q50: 0.025, q75: 0.045, q90: 0.075 } },
      { horizon_hours: 72, vix_direction_prob: 0.71, vix_up_prob: 0.48, vix_down_prob: 0.23, brent_return_quantiles: { q10: -0.01, q25: 0.015, q50: 0.032, q75: 0.055, q90: 0.09 } },
    ],
    top_drivers: [
      { feature: "Hormuz disruption", shap_value: 0.82, raw_value: 4.2, direction: "bearish" },
      { feature: "Trump policy-shock", shap_value: 0.65, raw_value: 3.1, direction: "bearish" },
      { feature: "Geopolitical tension", shap_value: 0.51, raw_value: 2.8, direction: "bearish" },
      { feature: "War tension level", shap_value: 0.44, raw_value: 6.0, direction: "bearish" },
      { feature: "Twitter fear index", shap_value: 0.38, raw_value: 0.6, direction: "bearish" },
      { feature: "Supply stress", shap_value: 0.21, raw_value: 0.4, direction: "bearish" },
      { feature: "Event decay signal", shap_value: -0.15, raw_value: 1.2, direction: "bullish" },
      { feature: "Elite-retail divergence", shap_value: -0.08, raw_value: -0.2, direction: "bullish" },
    ],
    feature_count: 132, processing_time_ms: 14.2,
    data_date_range: { start: "2024-01-01", end: "2024-06-30" },
  };
}
