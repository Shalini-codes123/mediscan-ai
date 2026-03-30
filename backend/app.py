import os, json, base64, io, hashlib, logging
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mediscan")

app = Flask(__name__)

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

MODEL_DIR = Path(__file__).parent / "models"
MODELS, META = {}, {}

# ─────────────────────────────────────────
# ✅ LOAD METRICS (NEW SOURCE OF TRUTH)
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
# ✅ LOAD MODELS (NO META FILE REQUIRED)
# ─────────────────────────────────────────
for name in ["diabetes","heart","kidney","pneumonia","brainTumor","skinCancer"]:
    pkl = MODEL_DIR / f"{name}.pkl"

    if pkl.exists():
        MODELS[name] = joblib.load(pkl)

        # Create fallback META dynamically
        META[name] = {
            "features": [],
            "metrics": ALL_METRICS.get(name, {})
        }

        logger.info(f"Loaded model: {name}")
    else:
        logger.warning(f"Model not found: {name}")

IMAGE_DISEASES = {"pneumonia","brainTumor","skinCancer"}

# ─────────────────────────────────────────
# IMAGE FEATURE EXTRACTION (unchanged)
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
    return np.pad(feat, (0, n-len(feat)))


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
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
        return jsonify({"error": "Model not found"}), 404

    model = MODELS[disease]
    stored_metrics = ALL_METRICS.get(disease, {})

    data = request.get_json(silent=True) or {}
    is_image = disease in IMAGE_DISEASES

    # ─────────────────────────────────────
    # IMAGE MODEL
    # ─────────────────────────────────────
    if is_image:
        b64 = data.get("image_b64", "")
        if not b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            img_bytes = base64.b64decode(b64)
        except:
            return jsonify({"error": "Invalid image"}), 400

        feat = image_to_features(img_bytes)
        feat = feat.reshape(1, -1)

    # ─────────────────────────────────────
    # TABULAR MODEL
    # ─────────────────────────────────────
    else:
        try:
            feat = np.array([list(map(float, data.values()))])
        except:
            return jsonify({"error": "Invalid input values"}), 400

    # ─────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────
    proba = model.predict_proba(feat)[0]
    pred = int(np.argmax(proba))

    return jsonify({
        "prediction": pred,
        "probability": float(proba[1]),
        "metrics": stored_metrics
    })


if __name__ == "__main__":
    app.run(debug=True)