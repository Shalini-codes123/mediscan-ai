import os, json, base64, io, binascii, logging
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mediscan")

app = Flask(__name__)

# ─────────────────────────────────────────
# CORS
# ─────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = os.getenv("CORS_ORIGIN", "*")
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/<path:path>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def options_handler(path=""):
    return "", 200


# ─────────────────────────────────────────
# 📦 MODEL DOWNLOAD
# ─────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

HF_REPO = os.getenv("HF_REPO")

def ensure_models():
    if any(MODEL_DIR.glob("*.pkl")):
        logger.info("Models already present")
        return

    if not HF_REPO:
        logger.warning("HF_REPO not set")
        return

    try:
        logger.info("Downloading models from Hugging Face...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO,
            repo_type="model",
            local_dir=str(MODEL_DIR)
        )

        logger.info("Models downloaded")

    except Exception as e:
        logger.error(f"Download failed: {e}")

ensure_models()


# ─────────────────────────────────────────
# LOAD METRICS + MODELS
# ─────────────────────────────────────────
ALL_METRICS = {}
metrics_file = MODEL_DIR / "metrics.json"

if metrics_file.exists():
    with open(metrics_file) as f:
        ALL_METRICS = json.load(f)

MODELS = {}
MODEL_NAMES = ["diabetes","heart","kidney","pneumonia","brainTumor","skinCancer"]

for name in MODEL_NAMES:
    pkl = MODEL_DIR / f"{name}.pkl"
    if pkl.exists():
        MODELS[name] = joblib.load(pkl)
        logger.info(f"Loaded model: {name}")
    else:
        logger.warning(f"Model not found: {name}")

IMAGE_DISEASES = {"pneumonia","brainTumor","skinCancer"}
FEATURE_ORDER = {
    "diabetes": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ],
    "heart": [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ],
    "kidney": [
        "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
        "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"
    ],
}
DISEASE_METADATA = {
    "diabetes": {
        "positive_label": "Positive",
        "negative_label": "Negative",
        "findings": {
            True: [
                "Glucose and metabolic markers align with a diabetes-positive pattern.",
                "The ensemble model recommends prompt physician follow-up for confirmatory testing.",
                "Lifestyle review and blood sugar monitoring should be prioritized."
            ],
            False: [
                "Submitted metabolic markers do not match the learned diabetes-positive pattern.",
                "Routine monitoring and preventive lifestyle habits remain advisable.",
                "This screening result should be interpreted alongside clinical evaluation."
            ],
        },
    },
    "heart": {
        "positive_label": "Positive",
        "negative_label": "Negative",
        "findings": {
            True: [
                "Cardiovascular features are consistent with elevated heart disease risk.",
                "Clinical correlation with ECG, lipid profile, and physician assessment is recommended.",
                "Risk-factor control should be reviewed without delay."
            ],
            False: [
                "Submitted cardiovascular markers do not strongly match the disease-positive pattern.",
                "Preventive cardiac care and regular screening are still recommended.",
                "Any persistent symptoms still warrant medical review."
            ],
        },
    },
    "kidney": {
        "positive_label": "Positive",
        "negative_label": "Negative",
        "findings": {
            True: [
                "Renal markers align with a chronic kidney disease-positive pattern.",
                "Kidney function review and repeat laboratory confirmation are advisable.",
                "Hydration, medication review, and nephrology guidance may be needed."
            ],
            False: [
                "Current renal markers do not strongly indicate kidney disease in this screening model.",
                "Ongoing monitoring is still important if symptoms or abnormal labs persist.",
                "Interpret this result together with creatinine, eGFR, and clinician review."
            ],
        },
    },
    "pneumonia": {
        "positive_label": "Detected",
        "negative_label": "Not Detected",
        "findings": {
            True: [
                "The uploaded chest image contains patterns associated with pneumonia.",
                "Radiology confirmation and symptom-based clinical assessment are recommended.",
                "Urgency increases if fever, hypoxia, or breathing difficulty is present."
            ],
            False: [
                "The uploaded chest image does not strongly match pneumonia-associated patterns.",
                "A normal AI screening result does not exclude early or subtle disease.",
                "Seek medical review if respiratory symptoms persist or worsen."
            ],
        },
    },
    "brainTumor": {
        "positive_label": "Detected",
        "negative_label": "Not Detected",
        "findings": {
            True: [
                "The MRI image shows features associated with a brain tumor-positive pattern.",
                "Formal radiology review and specialist referral are recommended.",
                "Neurologic symptoms should be treated as a higher-priority clinical concern."
            ],
            False: [
                "The MRI image does not strongly match the model's brain tumor-positive pattern.",
                "This does not replace expert radiology interpretation.",
                "Persistent neurologic symptoms still require clinical follow-up."
            ],
        },
    },
    "skinCancer": {
        "positive_label": "Detected",
        "negative_label": "Not Detected",
        "findings": {
            True: [
                "The lesion image contains features associated with skin cancer risk.",
                "Dermatology review and dermoscopic confirmation are recommended.",
                "Prompt evaluation is especially important for changing, bleeding, or irregular lesions."
            ],
            False: [
                "The lesion image does not strongly match the model's skin cancer-positive pattern.",
                "Benign-appearing lesions can still require review when clinically suspicious.",
                "Monitor for change in size, color, border, or symptoms."
            ],
        },
    },
}


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, round(float(value), 2)))


def build_prescription(disease, positive, risk_level):
    urgency_map = {"high": "urgent", "medium": "urgent", "low": "routine"}
    if disease in IMAGE_DISEASES and positive and risk_level == "high":
        urgency_map["high"] = "emergency"

    plans = {
        "diabetes": {
            True: {
                "immediateActions": [
                    "Schedule confirmatory HbA1c or fasting blood glucose testing.",
                    "Begin structured glucose monitoring and clinician review.",
                ],
                "medications": [
                    "Discuss initiation or adjustment of glucose-lowering therapy with a physician.",
                ],
                "lifestyle": [
                    "Reduce refined carbohydrates and improve meal consistency.",
                    "Aim for regular exercise and weight management where appropriate.",
                ],
                "followUp": "Repeat clinical review within 1-2 weeks or sooner if symptomatic.",
                "referral": "Endocrinology if control is poor or diagnosis is uncertain",
            },
            False: {
                "immediateActions": [
                    "Continue routine screening based on clinician advice and risk profile.",
                ],
                "medications": [],
                "lifestyle": [
                    "Maintain balanced diet, physical activity, and healthy body weight.",
                ],
                "followUp": "Routine follow-up during regular preventive care.",
                "referral": "None",
            },
        },
        "heart": {
            True: {
                "immediateActions": [
                    "Arrange clinician review of symptoms, ECG history, and risk factors.",
                    "Seek urgent care immediately for chest pain, syncope, or shortness of breath.",
                ],
                "medications": [
                    "Medication decisions should be made by a physician after clinical assessment.",
                ],
                "lifestyle": [
                    "Control blood pressure, cholesterol, smoking, diet, and exercise habits.",
                ],
                "followUp": "Cardiology or primary care review within 1 week.",
                "referral": "Cardiology",
            },
            False: {
                "immediateActions": [
                    "Continue preventive cardiovascular screening and risk-factor management.",
                ],
                "medications": [],
                "lifestyle": [
                    "Maintain exercise, heart-healthy diet, and blood pressure monitoring.",
                ],
                "followUp": "Routine preventive follow-up.",
                "referral": "None",
            },
        },
        "kidney": {
            True: {
                "immediateActions": [
                    "Review renal labs, hydration status, and medication nephrotoxicity.",
                ],
                "medications": [
                    "Medication adjustments should be reviewed with a clinician based on renal function.",
                ],
                "lifestyle": [
                    "Maintain hydration and follow renal-friendly dietary advice if prescribed.",
                ],
                "followUp": "Repeat renal function review within 1-2 weeks.",
                "referral": "Nephrology",
            },
            False: {
                "immediateActions": [
                    "Continue routine monitoring if risk factors or abnormal labs are present.",
                ],
                "medications": [],
                "lifestyle": [
                    "Maintain hydration, blood pressure control, and routine checkups.",
                ],
                "followUp": "Routine follow-up.",
                "referral": "None",
            },
        },
        "pneumonia": {
            True: {
                "immediateActions": [
                    "Arrange clinician or radiology review of the chest image promptly.",
                    "Seek urgent care for low oxygen, high fever, or breathing difficulty.",
                ],
                "medications": [
                    "Antibiotic choice should be made by a physician based on exam and imaging.",
                ],
                "lifestyle": [
                    "Rest, hydrate well, and monitor breathing symptoms closely.",
                ],
                "followUp": "Same-day or next-day medical review depending on symptoms.",
                "referral": "Pulmonology or emergency care if respiratory distress is present",
            },
            False: {
                "immediateActions": [
                    "Monitor symptoms and pursue clinician review if cough, fever, or dyspnea persists.",
                ],
                "medications": [],
                "lifestyle": [
                    "Rest, fluids, and symptom monitoring remain appropriate.",
                ],
                "followUp": "Follow up if symptoms worsen or do not improve.",
                "referral": "None",
            },
        },
        "brainTumor": {
            True: {
                "immediateActions": [
                    "Obtain formal radiology interpretation and urgent specialist review.",
                    "Seek urgent care for seizures, focal weakness, or rapid neurologic change.",
                ],
                "medications": [
                    "Treatment planning must be specialist-led after imaging confirmation.",
                ],
                "lifestyle": [
                    "Avoid delaying neurologic evaluation if symptoms are present.",
                ],
                "followUp": "Urgent specialist follow-up within 24-72 hours.",
                "referral": "Neurology or Neurosurgery",
            },
            False: {
                "immediateActions": [
                    "Continue formal radiology review if symptoms or MRI concerns remain.",
                ],
                "medications": [],
                "lifestyle": [
                    "Monitor neurologic symptoms and seek care if anything changes.",
                ],
                "followUp": "Routine or symptom-driven follow-up.",
                "referral": "None",
            },
        },
        "skinCancer": {
            True: {
                "immediateActions": [
                    "Arrange dermatology review and confirmatory lesion assessment.",
                ],
                "medications": [
                    "Do not self-treat suspicious lesions without dermatologist guidance.",
                ],
                "lifestyle": [
                    "Use sun protection and avoid further UV exposure to suspicious areas.",
                ],
                "followUp": "Dermatology follow-up within 1-2 weeks.",
                "referral": "Dermatology",
            },
            False: {
                "immediateActions": [
                    "Continue monitoring the lesion for change in color, size, or border.",
                ],
                "medications": [],
                "lifestyle": [
                    "Use regular sun protection and skin self-exams.",
                ],
                "followUp": "Routine follow-up if the lesion stays stable.",
                "referral": "None",
            },
        },
    }

    plan = plans[disease][positive].copy()
    plan["urgency"] = urgency_map[risk_level]
    return plan


def build_response(disease, positive_probability):
    positive = positive_probability >= 0.5
    probability_pct = clamp(positive_probability * 100)
    confidence_pct = clamp(max(positive_probability, 1 - positive_probability) * 100)

    if confidence_pct >= 85:
        risk_level = "high" if positive else "low"
    elif confidence_pct >= 65:
        risk_level = "medium" if positive else "low"
    else:
        risk_level = "medium"

    meta = DISEASE_METADATA[disease]
    label = meta["positive_label"] if positive else meta["negative_label"]

    return {
        "prediction": label,
        "probability": probability_pct,
        "confidence": confidence_pct,
        "riskLevel": risk_level,
        "findings": meta["findings"][positive],
        "metrics": ALL_METRICS.get(disease, {}),
        "prescription": build_prescription(disease, positive, risk_level),
        "disclaimer": (
            "This AI result is a screening aid and not a medical diagnosis. "
            "Always confirm with a qualified clinician."
        ),
    }


def parse_numeric_payload(disease, data):
    required = FEATURE_ORDER[disease]
    missing = [field for field in required if field not in data or data[field] in ("", None)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    try:
        return np.array([[float(data[field]) for field in required]])
    except (TypeError, ValueError):
        raise ValueError("All numeric inputs must be valid numbers")


def parse_image_payload(data):
    image_b64 = data.get("image_b64")
    if not image_b64:
        raise ValueError("image_b64 is required")

    try:
        img = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image_b64 must be valid raw base64 without a data URL prefix")

    if not img:
        raise ValueError("Decoded image is empty")

    return image_to_features(img).reshape(1, -1)


# ─────────────────────────────────────────
# IMAGE FEATURE EXTRACTION
# ─────────────────────────────────────────
def image_to_features(img_bytes):
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((64, 64))
        arr = np.array(img) / 255.0

        stats = [arr.mean(), arr.std(), arr.min(), arr.max()]
        hist,_ = np.histogram(arr, bins=64, range=(0,1))
        fft = np.abs(np.fft.rfft2(arr)).flatten()[:200]

        return np.concatenate([stats, hist, fft])

    except Exception as e:
        logger.error(e)
        return np.zeros(268)


# ─────────────────────────────────────────
# FRONTEND SERVING
# ─────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"

@app.route("/")
def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return send_from_directory(STATIC_DIR, "index.html")
    return jsonify({"message": "Frontend not built"})

@app.route("/<path:path>")
def serve_static(path):
    file = STATIC_DIR / path
    if file.exists():
        return send_from_directory(STATIC_DIR, path)
    return jsonify({"error": "Not found"}), 404


# ─────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(MODELS.keys())
    })

@app.route("/metrics")
def metrics():
    return jsonify(ALL_METRICS)

@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):

    if disease not in MODELS:
        return jsonify({"error": "Model not found"}), 404

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON"}), 400

    model = MODELS[disease]
    try:
        feat = parse_image_payload(data) if disease in IMAGE_DISEASES else parse_numeric_payload(disease, data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        proba = model.predict_proba(feat)[0]
        positive_probability = float(proba[1] if len(proba) > 1 else proba[0])
    except Exception as e:
        logger.exception("Prediction failed for %s", disease)
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify(build_response(disease, positive_probability))


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
