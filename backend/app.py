import os, json, base64, io, hashlib, logging
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

    model = MODELS[disease]
    data = request.get_json()

    if disease in IMAGE_DISEASES:
        img = base64.b64decode(data["image_b64"])
        feat = image_to_features(img).reshape(1, -1)
    else:
        feat = np.array([list(map(float, data.values()))])

    proba = model.predict_proba(feat)[0]

    return jsonify({
        "prediction": int(np.argmax(proba)),
        "probability": float(proba[1])
    })


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)