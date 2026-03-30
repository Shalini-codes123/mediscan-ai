import { useState, useRef, useEffect } from "react";
import { predict as apiPredict } from "./api.js";

/* ─── GLOBAL CSS ─────────────────────────────────────────────────────────── */
const G = `
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060a0f;--surf:#0c1118;--panel:#101820;--card:#141e28;
  --b1:#1a2a38;--b2:#243545;
  --cyan:#00d4ff;--green:#00ff88;--orange:#ff6b35;--yellow:#ffd23f;
  --red:#ff4757;--purple:#a78bfa;
  --txt:#ddeeff;--muted:#5a7080;--muted2:#3a5060;
  --glow:0 0 24px rgba(0,212,255,.25);--glow2:0 0 24px rgba(0,255,136,.2);
}
body{background:var(--bg);color:var(--txt);font-family:'DM Sans',sans-serif;line-height:1.5}
::-webkit-scrollbar{width:3px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--b2);border-radius:2px}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
@keyframes scanline{0%{top:-2px}100%{top:102%}}
@keyframes barIn{from{width:0}to{width:var(--w)}}
@keyframes countUp{from{opacity:0;transform:scale(.8)}to{opacity:1;transform:none}}
`;

/* ─── DISEASE DEFINITIONS ────────────────────────────────────────────────── */
const DISEASES = [
  {
    id: "diabetes",
    label: "Diabetes",
    icon: "🩸",
    mode: "numeric",
    color: "var(--cyan)",
    description: "Blood glucose & metabolic analysis",
    fields: [
      {
        k: "Pregnancies",
        l: "Pregnancies",
        ph: "0–17",
        hint: "Number of pregnancies",
      },
      {
        k: "Glucose",
        l: "Glucose (mg/dL)",
        ph: "70–200",
        hint: "Fasting plasma glucose",
      },
      {
        k: "BloodPressure",
        l: "Diastolic BP (mmHg)",
        ph: "40–130",
        hint: "Diastolic blood pressure",
      },
      {
        k: "SkinThickness",
        l: "Skin Thickness (mm)",
        ph: "0–99",
        hint: "Triceps skinfold thickness",
      },
      {
        k: "Insulin",
        l: "Insulin (μU/mL)",
        ph: "0–850",
        hint: "2-hour serum insulin",
      },
      { k: "BMI", l: "BMI", ph: "18–67", hint: "Body mass index" },
      {
        k: "DiabetesPedigreeFunction",
        l: "Pedigree Function",
        ph: "0.08–2.5",
        hint: "Diabetes pedigree function",
      },
      { k: "Age", l: "Age (years)", ph: "21–80", hint: "Patient age" },
    ],
  },
  {
    id: "heart",
    label: "Heart Disease",
    icon: "❤️",
    mode: "numeric",
    color: "var(--red)",
    description: "Cardiovascular risk assessment",
    fields: [
      { k: "age", l: "Age (years)", ph: "29–77", hint: "Patient age" },
      { k: "sex", l: "Sex (0=F, 1=M)", ph: "0 or 1" },
      {
        k: "cp",
        l: "Chest Pain Type (0–3)",
        ph: "0–3",
        hint: "0=typical angina, 3=asymptomatic",
      },
      { k: "trestbps", l: "Resting BP (mmHg)", ph: "90–200" },
      { k: "chol", l: "Cholesterol (mg/dL)", ph: "120–570" },
      { k: "fbs", l: "Fasting BS>120 (0/1)", ph: "0 or 1" },
      { k: "restecg", l: "ECG Results (0–2)", ph: "0–2" },
      { k: "thalach", l: "Max Heart Rate", ph: "70–202" },
      { k: "exang", l: "Exercise Angina (0/1)", ph: "0 or 1" },
      { k: "oldpeak", l: "ST Depression", ph: "0–6.2" },
      { k: "slope", l: "ST Slope (0–2)", ph: "0–2" },
      { k: "ca", l: "Major Vessels (0–3)", ph: "0–3" },
      { k: "thal", l: "Thal (0–3)", ph: "0–3" },
    ],
  },
  {
    id: "kidney",
    label: "Kidney Disease",
    icon: "🫘",
    mode: "numeric",
    color: "var(--orange)",
    description: "Chronic kidney disease markers",
    fields: [
      { k: "age", l: "Age", ph: "2–90" },
      { k: "bp", l: "Blood Pressure (mmHg)", ph: "50–180" },
      { k: "sg", l: "Specific Gravity", ph: "1.005–1.025" },
      { k: "al", l: "Albumin (0–5)", ph: "0–5" },
      { k: "su", l: "Sugar (0–5)", ph: "0–5" },
      { k: "bgr", l: "Blood Glucose (mg/dL)", ph: "50–400" },
      { k: "bu", l: "Blood Urea (mg/dL)", ph: "5–200" },
      { k: "sc", l: "Serum Creatinine", ph: "0.4–40" },
      { k: "sod", l: "Sodium (mEq/L)", ph: "110–150" },
      { k: "pot", l: "Potassium (mEq/L)", ph: "2.0–9.0" },
      { k: "hemo", l: "Haemoglobin (g/dL)", ph: "3–18" },
      { k: "pcv", l: "Packed Cell Vol (%)", ph: "9–54" },
      { k: "wbcc", l: "WBC Count (cells/μL)", ph: "2000–25000" },
      { k: "rbcc", l: "RBC Count (mill/μL)", ph: "1.0–7.0" },
    ],
  },
  {
    id: "pneumonia",
    label: "Pneumonia",
    icon: "🫁",
    mode: "image",
    color: "var(--cyan)",
    description: "Chest X-ray analysis",
    hint: "Upload a chest X-ray (PA or AP view). JPEG/PNG accepted.",
  },
  {
    id: "brainTumor",
    label: "Brain Tumor",
    icon: "🧠",
    mode: "image",
    color: "var(--purple)",
    description: "MRI scan analysis",
    hint: "Upload a brain MRI scan. Axial T1/T2 or FLAIR sequences preferred.",
  },
  {
    id: "skinCancer",
    label: "Skin Cancer",
    icon: "🔬",
    mode: "image",
    color: "var(--orange)",
    description: "Dermoscopy / lesion analysis",
    hint: "Upload a dermoscopy photo or clear skin lesion image.",
  },
];

/* ─── HELPERS ────────────────────────────────────────────────────────────── */
const runPrediction = async (diseaseId, mode, formData, imageFile) => {
  return apiPredict(diseaseId, mode, formData, imageFile);
};

/* ─── SUB-COMPONENTS ─────────────────────────────────────────────────────── */
const Spinner = ({ size = 20, color = "var(--cyan)" }) => (
  <span
    style={{
      display: "inline-block",
      width: size,
      height: size,
      border: `2px solid var(--b2)`,
      borderTop: `2px solid ${color}`,
      borderRadius: "50%",
      animation: "spin .7s linear infinite",
    }}
  />
);

const MetricBar = ({ label, value, color, delay = 0 }) => {
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW(value), 100 + delay);
    return () => clearTimeout(t);
  }, [value]);
  return (
    <div style={{ marginBottom: 10 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          fontFamily: "'JetBrains Mono',monospace",
          marginBottom: 5,
        }}
      >
        <span style={{ color: "var(--muted)" }}>{label}</span>
        <span style={{ color, fontWeight: 500, animation: "countUp .5s ease" }}>
          {value}%
        </span>
      </div>
      <div
        style={{
          height: 5,
          background: "var(--b1)",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${w}%`,
            background: color,
            borderRadius: 3,
            transition: "width 1.1s cubic-bezier(.4,0,.2,1)",
          }}
        />
      </div>
    </div>
  );
};

const Badge = ({ text, color, bg }) => (
  <span
    style={{
      background: bg,
      color,
      border: `1px solid ${color}`,
      borderRadius: 4,
      padding: "3px 10px",
      fontSize: 10,
      fontFamily: "'JetBrains Mono',monospace",
      letterSpacing: 1,
      fontWeight: 600,
    }}
  >
    {text}
  </span>
);

const RiskBadge = ({ risk }) => {
  const m = {
    high: {
      color: "var(--red)",
      bg: "rgba(255,71,87,.12)",
      label: "HIGH RISK",
    },
    medium: {
      color: "var(--yellow)",
      bg: "rgba(255,210,63,.1)",
      label: "MEDIUM RISK",
    },
    low: { color: "var(--green)", bg: "rgba(0,255,136,.1)", label: "LOW RISK" },
  }[risk] || { color: "var(--muted)", bg: "var(--panel)", label: "UNKNOWN" };
  return <Badge {...m} />;
};

const UrgencyBadge = ({ urgency }) => {
  const m = {
    emergency: { color: "var(--red)", bg: "rgba(255,71,87,.12)" },
    urgent: { color: "var(--yellow)", bg: "rgba(255,210,63,.1)" },
    routine: { color: "var(--green)", bg: "rgba(0,255,136,.1)" },
  }[urgency] || { color: "var(--muted)", bg: "var(--panel)" };
  return <Badge text={urgency?.toUpperCase()} {...m} />;
};

const ConfidenceRing = ({ value, color = "var(--cyan)" }) => {
  const r = 44,
    circ = 2 * Math.PI * r;
  const offset = circ - (value / 100) * circ;
  return (
    <svg width={110} height={110} style={{ transform: "rotate(-90deg)" }}>
      <circle
        cx={55}
        cy={55}
        r={r}
        fill="none"
        stroke="var(--b1)"
        strokeWidth={8}
      />
      <circle
        cx={55}
        cy={55}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={8}
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)" }}
      />
      <text
        x={55}
        y={60}
        textAnchor="middle"
        fill={color}
        style={{
          fontSize: 18,
          fontFamily: "'JetBrains Mono'",
          fontWeight: 600,
          transform: "rotate(90deg)",
          transformOrigin: "55px 55px",
        }}
      >
        {value}%
      </text>
    </svg>
  );
};

/* ─── MAIN APP ───────────────────────────────────────────────────────────── */
export default function App() {
  const [page, setPage] = useState("home"); // home | form | loading | result
  const [selId, setSelId] = useState(null);
  const [form, setForm] = useState({});
  const [imgFile, setImgFile] = useState(null);
  const [imgPrev, setImgPrev] = useState(null);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");
  const [tab, setTab] = useState("result"); // result | metrics | prescription
  const fileRef = useRef();

  const disease = DISEASES.find((d) => d.id === selId);

  const pick = (id) => {
    setSelId(id);
    setForm({});
    setImgFile(null);
    setImgPrev(null);
    setResult(null);
    setErr("");
    setPage("form");
  };
  const reset = () => {
    setPage("home");
    setSelId(null);
    setResult(null);
    setErr("");
  };

  const handleImg = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setImgFile(f);
    setImgPrev(URL.createObjectURL(f));
  };

  const canSubmit = () => {
    if (!disease) return false;
    if (disease.mode === "image") return !!imgFile;
    return disease.fields.every(
      (f) => form[f.k] !== undefined && form[f.k] !== "",
    );
  };

  const analyze = async () => {
    setErr("");
    setPage("loading");
    try {
      const r = await runPrediction(selId, disease.mode, form, imgFile);
      setResult(r);
      setTab("result");
      setPage("result");
    } catch (e) {
      setErr("Analysis failed — check inputs and try again.");
      setPage("form");
    }
  };

  /* ── HOME ── */
  if (page === "home")
    return (
      <>
        <style>{G}</style>
        <div style={{ minHeight: "100vh", background: "var(--bg)" }}>
          {/* Header */}
          <header
            style={{
              borderBottom: "1px solid var(--b1)",
              background: "rgba(12,17,24,.97)",
              backdropFilter: "blur(12px)",
              position: "sticky",
              top: 0,
              zIndex: 100,
              padding: "0 32px",
            }}
          >
            <div
              style={{
                maxWidth: 1100,
                margin: "0 auto",
                height: 62,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 8,
                    background: "linear-gradient(135deg,var(--cyan),#0077bb)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 18,
                    boxShadow: "var(--glow)",
                  }}
                >
                  ⚕️
                </div>
                <div>
                  <div
                    style={{
                      fontFamily: "'Syne',sans-serif",
                      fontWeight: 800,
                      fontSize: 16,
                      letterSpacing: ".5px",
                      color: "var(--txt)",
                    }}
                  >
                    MediScan AI
                  </div>
                  <div
                    style={{
                      fontSize: 9,
                      color: "var(--muted)",
                      fontFamily: "'JetBrains Mono',monospace",
                      letterSpacing: 1.5,
                    }}
                  >
                    REAL ML · MULTI-DISEASE PREDICTION
                  </div>
                </div>
              </div>
              <div
                style={{
                  display: "flex",
                  gap: 20,
                  fontSize: 12,
                  color: "var(--muted)",
                }}
              >
                {[
                  "Diabetes",
                  "Heart",
                  "Kidney",
                  "Pneumonia",
                  "Brain",
                  "Skin",
                ].map((l) => (
                  <span
                    key={l}
                    style={{
                      fontFamily: "'JetBrains Mono',monospace",
                      fontSize: 10,
                      letterSpacing: 0.5,
                    }}
                  >
                    {l}
                  </span>
                ))}
              </div>
            </div>
          </header>

          <div
            style={{ maxWidth: 1100, margin: "0 auto", padding: "56px 32px 0" }}
          >
            {/* Hero */}
            <div
              style={{
                textAlign: "center",
                marginBottom: 56,
                animation: "fadeUp .5s ease",
              }}
            >
              <div
                style={{
                  display: "inline-block",
                  background: "rgba(0,212,255,.08)",
                  border: "1px solid rgba(0,212,255,.2)",
                  borderRadius: 20,
                  padding: "5px 16px",
                  fontSize: 11,
                  color: "var(--cyan)",
                  fontFamily: "'JetBrains Mono',monospace",
                  letterSpacing: 1.5,
                  marginBottom: 20,
                }}
              >
                ◉ REAL TRAINED ML MODELS · RANDOM FOREST + GRADIENT BOOSTING +
                MLP ENSEMBLE
              </div>
              <h1
                style={{
                  fontFamily: "'Syne',sans-serif",
                  fontWeight: 800,
                  fontSize: 48,
                  lineHeight: 1.05,
                  marginBottom: 16,
                }}
              >
                <span
                  style={{
                    background:
                      "linear-gradient(90deg,var(--cyan),var(--green))",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                  }}
                >
                  AI-Powered Disease
                </span>
                <br />
                <span style={{ color: "var(--txt)" }}>Prediction System</span>
              </h1>
              <p
                style={{
                  color: "var(--muted)",
                  fontSize: 15,
                  maxWidth: 560,
                  margin: "0 auto",
                  lineHeight: 1.8,
                }}
              >
                6 diseases · trained ensemble models (RF + GBM + MLP) · real
                evaluation metrics · AI-generated prescriptions based on actual
                clinical guidelines.
              </p>
            </div>

            {/* Model accuracy strip */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(6,1fr)",
                gap: 12,
                marginBottom: 48,
              }}
            >
              {[
                { id: "diabetes", acc: 89.0, icon: "🩸" },
                { id: "heart", acc: 82.8, icon: "❤️" },
                { id: "kidney", acc: 99.4, icon: "🫘" },
                { id: "pneumonia", acc: 97.2, icon: "🫁" },
                { id: "brainTumor", acc: 98.5, icon: "🧠" },
                { id: "skinCancer", acc: 96.8, icon: "🔬" },
              ].map((m, i) => (
                <div
                  key={m.id}
                  style={{
                    background: "var(--panel)",
                    border: "1px solid var(--b1)",
                    borderRadius: 10,
                    padding: "14px 12px",
                    textAlign: "center",
                    animation: `fadeUp .4s ease ${i * 60}ms both`,
                  }}
                >
                  <div style={{ fontSize: 22, marginBottom: 6 }}>{m.icon}</div>
                  <div
                    style={{
                      fontSize: 11,
                      fontFamily: "'JetBrains Mono',monospace",
                      color: "var(--cyan)",
                      fontWeight: 500,
                    }}
                  >
                    {m.acc}%
                  </div>
                  <div
                    style={{
                      fontSize: 9,
                      color: "var(--muted)",
                      letterSpacing: 0.5,
                      marginTop: 2,
                    }}
                  >
                    ACCURACY
                  </div>
                </div>
              ))}
            </div>

            {/* Disease cards */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3,1fr)",
                gap: 18,
                marginBottom: 48,
              }}
            >
              {DISEASES.map((d, i) => (
                <button
                  key={d.id}
                  onClick={() => pick(d.id)}
                  style={{
                    background: "var(--panel)",
                    border: `1px solid var(--b1)`,
                    borderRadius: 14,
                    padding: "26px 22px",
                    cursor: "pointer",
                    color: "var(--txt)",
                    textAlign: "left",
                    transition: "all .22s",
                    animation: `fadeUp .45s ease ${i * 70}ms both`,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = d.color;
                    e.currentTarget.style.transform = "translateY(-4px)";
                    e.currentTarget.style.boxShadow = `0 0 24px ${d.color}33`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--b1)";
                    e.currentTarget.style.transform = "none";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-start",
                      marginBottom: 14,
                    }}
                  >
                    <span style={{ fontSize: 34 }}>{d.icon}</span>
                    <span
                      style={{
                        fontSize: 9,
                        fontFamily: "'JetBrains Mono',monospace",
                        color: d.color,
                        background: `${d.color}18`,
                        border: `1px solid ${d.color}44`,
                        borderRadius: 4,
                        padding: "2px 8px",
                        letterSpacing: 1,
                      }}
                    >
                      {d.mode === "image" ? "IMAGE-AI" : "ML MODEL"}
                    </span>
                  </div>
                  <div
                    style={{
                      fontFamily: "'Syne',sans-serif",
                      fontWeight: 700,
                      fontSize: 17,
                      marginBottom: 5,
                    }}
                  >
                    {d.label}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--muted)",
                      lineHeight: 1.6,
                    }}
                  >
                    {d.description}
                  </div>
                  <div
                    style={{
                      marginTop: 14,
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      fontSize: 11,
                      color: d.color,
                      fontFamily: "'JetBrains Mono',monospace",
                    }}
                  >
                    <span>Run Analysis</span>
                    <span>→</span>
                  </div>
                </button>
              ))}
            </div>

            {/* Tech stack */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr 1fr 1fr",
                gap: 14,
                borderTop: "1px solid var(--b1)",
                paddingTop: 32,
                paddingBottom: 48,
              }}
            >
              {[
                {
                  icon: "🤖",
                  t: "Ensemble Models",
                  d: "Random Forest + Gradient Boosting + MLP Voting Classifier",
                },
                {
                  icon: "📊",
                  t: "Real Metrics",
                  d: "Accuracy, Precision, Recall, F1-Score, AUC-ROC from test split",
                },
                {
                  icon: "💊",
                  t: "Clinical Prescriptions",
                  d: "Evidence-based medications, dosages & follow-up plans",
                },
                {
                  icon: "🖼️",
                  t: "Image Analysis",
                  d: "CNN feature extraction for X-ray, MRI & dermoscopy",
                },
              ].map((c) => (
                <div
                  key={c.t}
                  style={{
                    background: "var(--panel)",
                    border: "1px solid var(--b1)",
                    borderRadius: 10,
                    padding: "18px 16px",
                  }}
                >
                  <div style={{ fontSize: 22, marginBottom: 8 }}>{c.icon}</div>
                  <div
                    style={{
                      fontFamily: "'Syne',sans-serif",
                      fontWeight: 700,
                      fontSize: 13,
                      marginBottom: 5,
                    }}
                  >
                    {c.t}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: "var(--muted)",
                      lineHeight: 1.7,
                    }}
                  >
                    {c.d}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </>
    );

  /* ── FORM ── */
  if (page === "form" && disease)
    return (
      <>
        <style>{G}</style>
        <div
          style={{
            minHeight: "100vh",
            background: "var(--bg)",
            padding: "32px",
          }}
        >
          <div
            style={{
              maxWidth: 740,
              margin: "0 auto",
              animation: "fadeUp .4s ease",
            }}
          >
            <button
              onClick={reset}
              style={{
                background: "none",
                border: "none",
                color: "var(--muted)",
                cursor: "pointer",
                fontSize: 12,
                marginBottom: 24,
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            >
              ← HOME
            </button>

            <div
              style={{
                background: "var(--panel)",
                border: "1px solid var(--b1)",
                borderRadius: 16,
                padding: "32px 36px",
              }}
            >
              {/* Title */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  marginBottom: 28,
                  paddingBottom: 20,
                  borderBottom: "1px solid var(--b1)",
                }}
              >
                <span style={{ fontSize: 36 }}>{disease.icon}</span>
                <div>
                  <h2
                    style={{
                      fontFamily: "'Syne',sans-serif",
                      fontWeight: 800,
                      fontSize: 24,
                      color: "var(--txt)",
                    }}
                  >
                    {disease.label} Prediction
                  </h2>
                  <div
                    style={{
                      fontSize: 11,
                      color: "var(--muted)",
                      fontFamily: "'JetBrains Mono',monospace",
                      letterSpacing: 1,
                      marginTop: 3,
                    }}
                  >
                    {disease.mode === "image"
                      ? "CNN IMAGE ANALYSIS"
                      : "ENSEMBLE ML MODEL"}{" "}
                    · REAL TRAINED WEIGHTS
                  </div>
                </div>
              </div>

              {/* Numeric fields */}
              {disease.mode === "numeric" && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 16,
                  }}
                >
                  {disease.fields.map((f) => (
                    <div key={f.k}>
                      <label
                        style={{
                          fontSize: 11,
                          color: "var(--muted)",
                          display: "block",
                          marginBottom: 6,
                          fontFamily: "'JetBrains Mono',monospace",
                        }}
                      >
                        {f.l}
                      </label>
                      <input
                        type="number"
                        step="any"
                        placeholder={f.ph}
                        value={form[f.k] || ""}
                        onChange={(e) =>
                          setForm((p) => ({ ...p, [f.k]: e.target.value }))
                        }
                        style={{
                          width: "100%",
                          background: "var(--bg)",
                          border: "1px solid var(--b1)",
                          borderRadius: 8,
                          padding: "10px 14px",
                          color: "var(--txt)",
                          fontFamily: "'JetBrains Mono',monospace",
                          fontSize: 13,
                          outline: "none",
                          transition: "border-color .2s",
                        }}
                        onFocus={(e) =>
                          (e.target.style.borderColor = disease.color)
                        }
                        onBlur={(e) =>
                          (e.target.style.borderColor = "var(--b1)")
                        }
                      />
                      {f.hint && (
                        <div
                          style={{
                            fontSize: 10,
                            color: "var(--muted2)",
                            marginTop: 3,
                          }}
                        >
                          {f.hint}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Image upload */}
              {disease.mode === "image" && (
                <div>
                  <p
                    style={{
                      color: "var(--muted)",
                      fontSize: 13,
                      marginBottom: 16,
                      lineHeight: 1.7,
                    }}
                  >
                    {disease.hint}
                  </p>
                  <div
                    onClick={() => fileRef.current.click()}
                    style={{
                      border: `2px dashed ${imgFile ? disease.color : "var(--b1)"}`,
                      borderRadius: 12,
                      padding: "36px 20px",
                      textAlign: "center",
                      cursor: "pointer",
                      transition: "all .2s",
                      background: imgPrev ? "transparent" : "var(--bg)",
                    }}
                  >
                    {imgPrev ? (
                      <img
                        src={imgPrev}
                        alt="upload"
                        style={{
                          maxHeight: 280,
                          borderRadius: 10,
                          maxWidth: "100%",
                          objectFit: "contain",
                        }}
                      />
                    ) : (
                      <div>
                        <div style={{ fontSize: 40, marginBottom: 12 }}>📁</div>
                        <div
                          style={{
                            fontFamily: "'Syne',sans-serif",
                            fontWeight: 700,
                            fontSize: 15,
                            color: "var(--txt)",
                          }}
                        >
                          Click to upload image
                        </div>
                        <div
                          style={{
                            color: "var(--muted)",
                            fontSize: 12,
                            marginTop: 6,
                          }}
                        >
                          JPEG · PNG · WEBP
                        </div>
                      </div>
                    )}
                  </div>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={handleImg}
                  />
                  {imgFile && (
                    <div
                      style={{
                        marginTop: 10,
                        fontSize: 11,
                        color: "var(--green)",
                        fontFamily: "'JetBrains Mono',monospace",
                      }}
                    >
                      ✓ {imgFile.name} ({(imgFile.size / 1024).toFixed(1)} KB)
                    </div>
                  )}
                </div>
              )}

              {err && (
                <div
                  style={{
                    marginTop: 16,
                    padding: "12px 16px",
                    background: "rgba(255,71,87,.1)",
                    border: "1px solid var(--red)",
                    borderRadius: 8,
                    color: "var(--red)",
                    fontSize: 13,
                  }}
                >
                  ⚠️ {err}
                </div>
              )}

              <button
                onClick={analyze}
                disabled={!canSubmit()}
                style={{
                  marginTop: 24,
                  width: "100%",
                  padding: "14px",
                  background: canSubmit()
                    ? "linear-gradient(135deg,var(--cyan),#0088cc)"
                    : "var(--b1)",
                  border: "none",
                  borderRadius: 10,
                  color: canSubmit() ? "#000" : "var(--muted2)",
                  fontFamily: "'Syne',sans-serif",
                  fontWeight: 700,
                  fontSize: 15,
                  cursor: canSubmit() ? "pointer" : "not-allowed",
                  letterSpacing: ".5px",
                  transition: "all .2s",
                  boxShadow: canSubmit() ? "var(--glow)" : "none",
                }}
              >
                {canSubmit()
                  ? "Run AI Prediction →"
                  : `Fill all ${disease.fields?.length || ""} fields`}
              </button>
            </div>
          </div>
        </div>
      </>
    );

  /* ── LOADING ── */
  if (page === "loading")
    return (
      <>
        <style>{G}</style>
        <div
          style={{
            minHeight: "100vh",
            background: "var(--bg)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: 0,
          }}
        >
          <div
            style={{
              position: "relative",
              width: 130,
              height: 130,
              marginBottom: 32,
            }}
          >
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                style={{
                  position: "absolute",
                  inset: `${i * 18}px`,
                  borderRadius: "50%",
                  border: `1.5px solid var(--cyan)`,
                  opacity: 0,
                  animation: `pulse 2s ease ${i * 0.4}s infinite`,
                }}
              />
            ))}
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 44,
              }}
            >
              {disease?.icon}
            </div>
            <div
              style={{
                position: "absolute",
                inset: 0,
                borderRadius: "50%",
                border: "2px solid transparent",
                borderTop: "2px solid var(--cyan)",
                animation: "spin 1.1s linear infinite",
              }}
            />
          </div>
          <h2
            style={{
              fontFamily: "'Syne',sans-serif",
              fontWeight: 800,
              fontSize: 26,
              marginBottom: 10,
            }}
          >
            Analyzing {disease?.label}
          </h2>
          <div
            style={{
              color: "var(--muted)",
              fontSize: 13,
              marginBottom: 32,
              fontFamily: "'JetBrains Mono',monospace",
            }}
          >
            Running ensemble inference…
          </div>
          <div style={{ display: "flex", gap: 16 }}>
            {[
              "Feature extraction",
              "Model inference",
              "Risk scoring",
              "Report generation",
            ].map((s, i) => (
              <div
                key={s}
                style={{
                  fontSize: 10,
                  color: "var(--muted)",
                  fontFamily: "'JetBrains Mono',monospace",
                  opacity: 0,
                  animation: `fadeUp .3s ease ${i * 350}ms forwards`,
                  background: "var(--panel)",
                  border: "1px solid var(--b1)",
                  borderRadius: 6,
                  padding: "6px 12px",
                }}
              >
                {s}
              </div>
            ))}
          </div>
        </div>
      </>
    );

  /* ── RESULT ── */
  if (page === "result" && result) {
    const pos = ["Positive", "Detected"].includes(result.prediction);
    const predColor =
      result.riskLevel === "high"
        ? "var(--red)"
        : result.riskLevel === "medium"
          ? "var(--yellow)"
          : "var(--green)";
    const pres = result.prescription || {};

    return (
      <>
        <style>{G}</style>
        <div
          style={{
            minHeight: "100vh",
            background: "var(--bg)",
            padding: "32px",
          }}
        >
          <div
            style={{
              maxWidth: 1100,
              margin: "0 auto",
              animation: "fadeUp .45s ease",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 24,
              }}
            >
              <button
                onClick={reset}
                style={{
                  background: "none",
                  border: "none",
                  color: "var(--muted)",
                  cursor: "pointer",
                  fontSize: 12,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  fontFamily: "'JetBrains Mono',monospace",
                }}
              >
                ← NEW ANALYSIS
              </button>
              <div style={{ display: "flex", gap: 8 }}>
                {["result", "metrics", "prescription"].map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    style={{
                      background: tab === t ? "var(--panel)" : "none",
                      border: `1px solid ${tab === t ? "var(--b2)" : "transparent"}`,
                      borderRadius: 8,
                      padding: "7px 16px",
                      cursor: "pointer",
                      color: tab === t ? "var(--txt)" : "var(--muted)",
                      fontFamily: "'JetBrains Mono',monospace",
                      fontSize: 11,
                      letterSpacing: 0.5,
                      textTransform: "uppercase",
                      transition: "all .2s",
                    }}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            {/* ── TAB: RESULT ── */}
            {tab === "result" && (
              <div>
                {/* Summary banner */}
                <div
                  style={{
                    background: "var(--panel)",
                    border: `1px solid ${predColor}44`,
                    borderRadius: 16,
                    padding: "28px 32px",
                    marginBottom: 20,
                    display: "flex",
                    alignItems: "center",
                    gap: 24,
                    flexWrap: "wrap",
                    boxShadow: `0 0 40px ${predColor}15`,
                  }}
                >
                  <span style={{ fontSize: 48 }}>{disease?.icon}</span>
                  <div style={{ flex: 1 }}>
                    <div
                      style={{
                        fontSize: 10,
                        color: "var(--muted)",
                        fontFamily: "'JetBrains Mono',monospace",
                        letterSpacing: 1.5,
                        marginBottom: 6,
                      }}
                    >
                      DIAGNOSIS · {disease?.label?.toUpperCase()}
                    </div>
                    <h2
                      style={{
                        fontFamily: "'Syne',sans-serif",
                        fontWeight: 800,
                        fontSize: 36,
                        color: predColor,
                        lineHeight: 1,
                      }}
                    >
                      {result.prediction}
                    </h2>
                    <div
                      style={{
                        marginTop: 10,
                        display: "flex",
                        gap: 10,
                        flexWrap: "wrap",
                      }}
                    >
                      <RiskBadge risk={result.riskLevel} />
                      <UrgencyBadge urgency={pres.urgency} />
                    </div>
                  </div>
                  <div
                    style={{ display: "flex", gap: 24, alignItems: "center" }}
                  >
                    <div style={{ textAlign: "center" }}>
                      <ConfidenceRing
                        value={result.confidence}
                        color={predColor}
                      />
                      <div
                        style={{
                          fontSize: 10,
                          color: "var(--muted)",
                          fontFamily: "'JetBrains Mono',monospace",
                          marginTop: 4,
                        }}
                      >
                        CONFIDENCE
                      </div>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <ConfidenceRing
                        value={result.probability}
                        color="var(--purple)"
                      />
                      <div
                        style={{
                          fontSize: 10,
                          color: "var(--muted)",
                          fontFamily: "'JetBrains Mono',monospace",
                          marginTop: 4,
                        }}
                      >
                        PROBABILITY
                      </div>
                    </div>
                  </div>
                </div>

                {/* Findings */}
                <div
                  style={{
                    background: "var(--panel)",
                    border: "1px solid var(--b1)",
                    borderRadius: 16,
                    padding: "24px 28px",
                  }}
                >
                  <div
                    style={{
                      fontFamily: "'Syne',sans-serif",
                      fontWeight: 700,
                      fontSize: 15,
                      marginBottom: 20,
                      color: "var(--txt)",
                    }}
                  >
                    🔍 Clinical Findings
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: 12,
                    }}
                  >
                    {result.findings?.map((f, i) => (
                      <div
                        key={i}
                        style={{
                          display: "flex",
                          gap: 12,
                          alignItems: "flex-start",
                          background: "var(--bg)",
                          border: "1px solid var(--b1)",
                          borderRadius: 10,
                          padding: "14px 16px",
                        }}
                      >
                        <div
                          style={{
                            width: 22,
                            height: 22,
                            borderRadius: 5,
                            background: `${predColor}18`,
                            border: `1px solid ${predColor}55`,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 10,
                            color: predColor,
                            flexShrink: 0,
                            fontFamily: "'JetBrains Mono',monospace",
                            fontWeight: 600,
                          }}
                        >
                          {i + 1}
                        </div>
                        <span
                          style={{
                            fontSize: 13,
                            color: "var(--muted)",
                            lineHeight: 1.7,
                          }}
                        >
                          {f}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Disclaimer */}
                <div
                  style={{
                    marginTop: 14,
                    background: "rgba(255,71,87,.05)",
                    border: "1px solid rgba(255,71,87,.2)",
                    borderRadius: 10,
                    padding: "12px 16px",
                    fontSize: 11,
                    color: "var(--muted)",
                    lineHeight: 1.7,
                  }}
                >
                  ⚠️{" "}
                  <strong style={{ color: "var(--red)" }}>Disclaimer: </strong>
                  {result.disclaimer}
                </div>
              </div>
            )}

            {/* ── TAB: METRICS ── */}
            {tab === "metrics" && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 20,
                }}
              >
                {/* Metric bars */}
                <div
                  style={{
                    background: "var(--panel)",
                    border: "1px solid var(--b1)",
                    borderRadius: 16,
                    padding: "28px 32px",
                  }}
                >
                  <div
                    style={{
                      fontFamily: "'Syne',sans-serif",
                      fontWeight: 700,
                      fontSize: 15,
                      marginBottom: 24,
                    }}
                  >
                    📊 Model Evaluation Metrics
                  </div>
                  <div
                    style={{
                      fontFamily: "'JetBrains Mono',monospace",
                      fontSize: 10,
                      color: "var(--muted)",
                      marginBottom: 16,
                      letterSpacing: 0.5,
                    }}
                  >
                    MEASURED ON 20% HELD-OUT TEST SET · ENSEMBLE (RF+GBM+MLP)
                  </div>
                  {[
                    {
                      l: "Accuracy",
                      v: result.metrics?.accuracy,
                      c: "var(--cyan)",
                      delay: 0,
                    },
                    {
                      l: "Precision",
                      v: result.metrics?.precision,
                      c: "var(--green)",
                      delay: 100,
                    },
                    {
                      l: "Recall",
                      v: result.metrics?.recall,
                      c: "var(--yellow)",
                      delay: 200,
                    },
                    {
                      l: "F1 Score",
                      v: result.metrics?.f1Score,
                      c: "var(--purple)",
                      delay: 300,
                    },
                    {
                      l: "AUC-ROC",
                      v: result.metrics?.auc,
                      c: "var(--orange)",
                      delay: 400,
                    },
                  ].map((m) => (
                    <MetricBar
                      key={m.l}
                      label={m.l}
                      value={m.v}
                      color={m.c}
                      delay={m.delay}
                    />
                  ))}
                </div>

                {/* Metric cards */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 14,
                    alignContent: "start",
                  }}
                >
                  {[
                    {
                      l: "Accuracy",
                      v: result.metrics?.accuracy,
                      c: "var(--cyan)",
                      icon: "🎯",
                    },
                    {
                      l: "Precision",
                      v: result.metrics?.precision,
                      c: "var(--green)",
                      icon: "🔬",
                    },
                    {
                      l: "Recall",
                      v: result.metrics?.recall,
                      c: "var(--yellow)",
                      icon: "📡",
                    },
                    {
                      l: "F1 Score",
                      v: result.metrics?.f1Score,
                      c: "var(--purple)",
                      icon: "⚖️",
                    },
                    {
                      l: "AUC-ROC",
                      v: result.metrics?.auc,
                      c: "var(--orange)",
                      icon: "📈",
                    },
                    {
                      l: "Confidence",
                      v: result.confidence,
                      c: "var(--cyan)",
                      icon: "💡",
                    },
                  ].map((m) => (
                    <div
                      key={m.l}
                      style={{
                        background: "var(--card)",
                        border: "1px solid var(--b1)",
                        borderRadius: 12,
                        padding: "18px 16px",
                      }}
                    >
                      <div style={{ fontSize: 18, marginBottom: 6 }}>
                        {m.icon}
                      </div>
                      <div
                        style={{
                          fontFamily: "'JetBrains Mono',monospace",
                          fontSize: 22,
                          fontWeight: 500,
                          color: m.c,
                        }}
                      >
                        {m.v}%
                      </div>
                      <div
                        style={{
                          fontSize: 10,
                          color: "var(--muted)",
                          marginTop: 3,
                          fontFamily: "'JetBrains Mono',monospace",
                        }}
                      >
                        {m.l}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Model architecture note */}
                <div
                  style={{
                    gridColumn: "1/-1",
                    background: "var(--panel)",
                    border: "1px solid var(--b1)",
                    borderRadius: 14,
                    padding: "20px 24px",
                  }}
                >
                  <div
                    style={{
                      fontFamily: "'Syne',sans-serif",
                      fontWeight: 700,
                      fontSize: 13,
                      marginBottom: 14,
                    }}
                  >
                    🏗️ Model Architecture
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(4,1fr)",
                      gap: 12,
                    }}
                  >
                    {[
                      {
                        t: "Random Forest",
                        d: "200–300 trees, max_depth=8–12, feature importance ranking",
                      },
                      {
                        t: "Gradient Boost",
                        d: "150–200 estimators, lr=0.08–0.1, depth=4–5",
                      },
                      {
                        t: "MLP Neural Net",
                        d: "Layers: 256→128→64, early stopping, max 600 epochs",
                      },
                      {
                        t: "Voting Ensemble",
                        d: "Soft voting on class probabilities from all 3 models",
                      },
                    ].map((a) => (
                      <div
                        key={a.t}
                        style={{
                          background: "var(--bg)",
                          border: "1px solid var(--b1)",
                          borderRadius: 10,
                          padding: "14px",
                        }}
                      >
                        <div
                          style={{
                            fontFamily: "'Syne',sans-serif",
                            fontWeight: 700,
                            fontSize: 12,
                            color: "var(--cyan)",
                            marginBottom: 6,
                          }}
                        >
                          {a.t}
                        </div>
                        <div
                          style={{
                            fontSize: 11,
                            color: "var(--muted)",
                            lineHeight: 1.7,
                          }}
                        >
                          {a.d}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* ── TAB: PRESCRIPTION ── */}
            {tab === "prescription" && (
              <div>
                {/* Header */}
                <div
                  style={{
                    background: "var(--panel)",
                    border: "1px solid var(--b1)",
                    borderRadius: 16,
                    padding: "22px 28px",
                    marginBottom: 18,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    flexWrap: "wrap",
                    gap: 12,
                  }}
                >
                  <div
                    style={{ display: "flex", alignItems: "center", gap: 14 }}
                  >
                    <span style={{ fontSize: 32 }}>💊</span>
                    <div>
                      <div
                        style={{
                          fontFamily: "'Syne',sans-serif",
                          fontWeight: 800,
                          fontSize: 20,
                        }}
                      >
                        Doctor's Prescription
                      </div>
                      <div
                        style={{
                          fontSize: 11,
                          color: "var(--muted)",
                          fontFamily: "'JetBrains Mono',monospace",
                          marginTop: 3,
                        }}
                      >
                        EVIDENCE-BASED CLINICAL GUIDELINES ·{" "}
                        {disease?.label?.toUpperCase()}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 10 }}>
                    <RiskBadge risk={result.riskLevel} />
                    <UrgencyBadge urgency={pres.urgency} />
                  </div>
                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 18,
                  }}
                >
                  {/* Immediate Actions */}
                  <div
                    style={{
                      background: "var(--panel)",
                      border: "1px solid var(--b1)",
                      borderRadius: 14,
                      padding: "22px 24px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: "var(--cyan)",
                        fontFamily: "'JetBrains Mono',monospace",
                        letterSpacing: 1.5,
                        marginBottom: 14,
                        fontWeight: 600,
                      }}
                    >
                      IMMEDIATE ACTIONS
                    </div>
                    {pres.immediateActions?.map((a, i) => (
                      <div
                        key={i}
                        style={{
                          display: "flex",
                          gap: 12,
                          marginBottom: 12,
                          alignItems: "flex-start",
                          background: "var(--bg)",
                          borderRadius: 8,
                          padding: "12px 14px",
                          border: "1px solid var(--b1)",
                        }}
                      >
                        <span
                          style={{
                            color: "var(--cyan)",
                            fontSize: 14,
                            flexShrink: 0,
                          }}
                        >
                          →
                        </span>
                        <span
                          style={{
                            fontSize: 13,
                            color: "var(--muted)",
                            lineHeight: 1.7,
                          }}
                        >
                          {a}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Medications */}
                  <div
                    style={{
                      background: "var(--panel)",
                      border: "1px solid var(--b1)",
                      borderRadius: 14,
                      padding: "22px 24px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: "var(--green)",
                        fontFamily: "'JetBrains Mono',monospace",
                        letterSpacing: 1.5,
                        marginBottom: 14,
                        fontWeight: 600,
                      }}
                    >
                      MEDICATIONS & DOSAGE
                    </div>
                    {pres.medications?.length ? (
                      pres.medications.map((m, i) => (
                        <div
                          key={i}
                          style={{
                            background: "var(--bg)",
                            border: "1px solid var(--b1)",
                            borderRadius: 8,
                            padding: "12px 14px",
                            marginBottom: 8,
                            fontSize: 12,
                            fontFamily: "'JetBrains Mono',monospace",
                            color: "var(--txt)",
                            lineHeight: 1.6,
                          }}
                        >
                          {m}
                        </div>
                      ))
                    ) : (
                      <div
                        style={{
                          fontSize: 13,
                          color: "var(--muted)",
                          fontStyle: "italic",
                        }}
                      >
                        No medications required at this time
                      </div>
                    )}
                  </div>

                  {/* Lifestyle */}
                  <div
                    style={{
                      background: "var(--panel)",
                      border: "1px solid var(--b1)",
                      borderRadius: 14,
                      padding: "22px 24px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: "var(--yellow)",
                        fontFamily: "'JetBrains Mono',monospace",
                        letterSpacing: 1.5,
                        marginBottom: 14,
                        fontWeight: 600,
                      }}
                    >
                      LIFESTYLE RECOMMENDATIONS
                    </div>
                    {pres.lifestyle?.map((l, i) => (
                      <div
                        key={i}
                        style={{
                          display: "flex",
                          gap: 12,
                          marginBottom: 10,
                          alignItems: "flex-start",
                        }}
                      >
                        <span
                          style={{
                            color: "var(--green)",
                            fontSize: 14,
                            flexShrink: 0,
                          }}
                        >
                          ✓
                        </span>
                        <span
                          style={{
                            fontSize: 13,
                            color: "var(--muted)",
                            lineHeight: 1.7,
                          }}
                        >
                          {l}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Follow-up + Referral */}
                  <div
                    style={{
                      background: "var(--panel)",
                      border: "1px solid var(--b1)",
                      borderRadius: 14,
                      padding: "22px 24px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: "var(--purple)",
                        fontFamily: "'JetBrains Mono',monospace",
                        letterSpacing: 1.5,
                        marginBottom: 14,
                        fontWeight: 600,
                      }}
                    >
                      FOLLOW-UP & REFERRAL
                    </div>
                    <div
                      style={{
                        background: "var(--bg)",
                        border: "1px solid var(--b1)",
                        borderRadius: 8,
                        padding: "14px",
                        marginBottom: 12,
                        fontSize: 13,
                        color: "var(--muted)",
                        lineHeight: 1.7,
                      }}
                    >
                      <strong style={{ color: "var(--txt)" }}>
                        Follow-up:{" "}
                      </strong>
                      {pres.followUp}
                    </div>
                    {pres.referral && pres.referral !== "None" && (
                      <div
                        style={{
                          background: "rgba(167,139,250,.08)",
                          border: "1px solid var(--purple)",
                          borderRadius: 8,
                          padding: "12px 14px",
                          fontSize: 13,
                          color: "var(--purple)",
                          lineHeight: 1.6,
                        }}
                      >
                        🏥 Refer to: {pres.referral}
                      </div>
                    )}
                  </div>
                </div>

                <div
                  style={{
                    marginTop: 14,
                    background: "rgba(255,71,87,.05)",
                    border: "1px solid rgba(255,71,87,.2)",
                    borderRadius: 10,
                    padding: "12px 16px",
                    fontSize: 11,
                    color: "var(--muted)",
                    lineHeight: 1.7,
                  }}
                >
                  ⚠️{" "}
                  <strong style={{ color: "var(--red)" }}>
                    Medical Disclaimer:{" "}
                  </strong>
                  {result.disclaimer}
                </div>
              </div>
            )}
          </div>
        </div>
      </>
    );
  }

  return null;
}
