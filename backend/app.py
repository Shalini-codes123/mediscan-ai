import os, json, base64, io, hashlib, logging
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, request, jsonify

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
# 📦 MODEL DOWNLOAD (CRITICAL FIX)
# ─────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

HF_REPO = os.getenv("HF_REPO")  # e.g. "yourusername/mediscan-models"

def ensure_models():
    # If models already exist → skip
    if any(MODEL_DIR.glob("*.pkl")):
        logger.info("Models already present")
        return

    if not HF_REPO:
        logger.warning("HF_REPO not set and no local models found")
        return

    try:
        logger.info("Downloading models from Hugging Face...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=HF_REPO,
            repo_type="model",
            local_dir=str(MODEL_DIR)
        )

        logger.info("Models downloaded successfully")

    except Exception as e:
        logger.error(f"Model download failed: {e}")


# Call before loading
ensure_models()


# ─────────────────────────────────────────
# LOAD METRICS
# ─────────────────────────────────────────
METRICS_FILE = MODEL_DIR / "metrics.json"
ALL_METRICS = {}

if METRICS_FILE.exists():
    with open(METRICS_FILE) as f:
        ALL_METRICS = json.load(f)
    logger.info("Loaded metrics.json")
else:
    logger.warning("metrics.json not found")


# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
MODELS = {}

MODEL_NAMES = ["diabetes","heart","kidney","pneumonia","brainTumor","skinCancer"]

for name in MODEL_NAMES:
    pkl = MODEL_DIR / f"{name}.pkl"

    if pkl.exists():
        try:
            MODELS[name] = joblib.load(pkl)
            logger.info(f"Loaded model: {name}")
        except Exception as e:
            logger.error(f"Failed loading {name}: {e}")
    else:
        logger.warning(f"Model not found: {name}")


IMAGE_DISEASES = {"pneumonia","brainTumor","skinCancer"}


# ─────────────────────────────────────────
# IMAGE FEATURE EXTRACTION
# ─────────────────────────────────────────
def image_to_features(img_bytes: bytes) -> np.ndarray:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0

        stats = [arr.mean(), arr.std(), arr.min(), arr.max(),
                 np.percentile(arr,10), np.percentile(arr,25),
                 np.percentile(arr,75), np.percentile(arr,90)]

        hist,_ = np.histogram(arr, bins=64, range=(0,1))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-9)

        fft = np.abs(np.fft.rfft2(arr)).flatten()[:200]

        blocks = []
        for i in range(0,64,16):
            for j in range(0,64,16):
                blocks.append(arr[i:i+16,j:j+16].mean())

        return np.concatenate([stats, hist, fft, blocks])

    except Exception as e:
        logger.error(f"Image feature extraction failed: {e}")

        h = int(hashlib.md5(img_bytes).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        return rng.normal(0.5, 0.15, 288).astype(np.float32)


def pad_or_trim(feat, n):
    if len(feat) >= n:
        return feat[:n]
    return np.pad(feat, (0, n - len(feat)))


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "message": "MediScan API is running",
        "endpoints": [
            "/health",
            "/metrics",
            "/predict/<disease>"
        ]
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(MODELS.keys())
    })


@app.route("/metrics")
def all_metrics():
    return jsonify(ALL_METRICS)


@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):

    if disease not in MODELS:
        return jsonify({
            "error": f"Model '{disease}' not loaded",
            "available_models": list(MODELS.keys())
        }), 404

    model = MODELS[disease]
    stored_metrics = ALL_METRICS.get(disease, {})

    data = request.get_json(silent=True) or {}
    is_image = disease in IMAGE_DISEASES

    # IMAGE
    if is_image:
        b64 = data.get("image_b64", "")
        if not b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            img_bytes = base64.b64decode(b64)
        except:
            return jsonify({"error": "Invalid image"}), 400

        feat = image_to_features(img_bytes).reshape(1, -1)

    # TABULAR
    else:
        try:
            feat = np.array([list(map(float, data.values()))])
        except:
            return jsonify({"error": "Invalid input values"}), 400

    # PREDICT
    proba = model.predict_proba(feat)[0]
    pred = int(np.argmax(proba))

    return jsonify({
        "prediction": pred,
        "probability": float(proba[1]),
        "metrics": stored_metrics
    })


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)