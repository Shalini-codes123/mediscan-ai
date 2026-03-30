import os, sys, json, warnings, shutil
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR= ROOT / "backend" / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# KAGGLE AUTH
# ─────────────────────────────────────────
def check_kaggle():
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle key missing at ~/.kaggle/kaggle.json")
        sys.exit(1)
    api = KaggleApi()
    api.authenticate()
    print("✓ Kaggle ready")

def kaggle_download(dataset, path):
    from kaggle.api.kaggle_api_extended import KaggleApi
    dest = DATA_DIR / path
    dest.mkdir(parents=True, exist_ok=True)
    # Skip if already downloaded
    existing = list(dest.iterdir()) if dest.exists() else []
    if existing:
        print(f"  ✓ Already downloaded: {path}")
        return
    print(f"  Downloading {dataset}...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(dest), unzip=True)
    print(f"  ✓ Done")

# ─────────────────────────────────────────
# ML IMPORTS
# ─────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.impute import SimpleImputer

ALL_METRICS = {}

def make_model():
    return VotingClassifier([
        ("rf",  RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=42, n_jobs=-1)),
        ("gb",  GradientBoostingClassifier(n_estimators=150, random_state=42)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ], voting="soft")

def evaluate_and_save(name, pipe, X_test, y_test, feature_names):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred)                   * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0)    * 100, 2),
        "f1Score":   round(f1_score(y_test, y_pred, zero_division=0)        * 100, 2),
        "auc":       round(roc_auc_score(y_test, y_prob)                    * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"  [{name}] Acc:{metrics['accuracy']}% | Prec:{metrics['precision']}% | "
          f"Rec:{metrics['recall']}% | F1:{metrics['f1Score']}% | AUC:{metrics['auc']}%")

    joblib.dump(pipe, MODEL_DIR / f"{name}.pkl")
    with open(MODEL_DIR / f"{name}_meta.json", "w") as f:
        json.dump({"features": feature_names, "metrics": metrics}, f, indent=2)

    ALL_METRICS[name] = metrics

# ─────────────────────────────────────────
# TABULAR TRAINING
# ─────────────────────────────────────────
def train_tabular(name, df, target):
    if target not in df.columns:
        print("  ⚠ Target column missing"); return

    print(f"  Shape: {df.shape}")
    y    = df[target].copy()
    X_df = df.drop(columns=[target]).copy()

    for col in X_df.columns:
        if X_df[col].dtype == object:
            X_df[col] = LabelEncoder().fit_transform(
                X_df[col].astype(str).str.strip())

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df.fillna(X_df.median(), inplace=True)

    mask = y.notna()
    X_df, y = X_df[mask], y[mask]

    if len(X_df) < 20 or len(np.unique(y)) < 2:
        print("  ⚠ Dataset too small or single class. Skipping."); return

    feat_names = list(X_df.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df.values, y.values, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("imp",   SimpleImputer()),
        ("sc",    StandardScaler()),
        ("model", make_model())
    ])
    pipe.fit(X_tr, y_tr)
    evaluate_and_save(name, pipe, X_te, y_te, feat_names)

# ─────────────────────────────────────────
# IMAGE FEATURE EXTRACTION
# ─────────────────────────────────────────
def extract_features_from_folder(folder: Path, label: int, max_images=1200):
    """
    Extract 288-dim feature vector per image.
    Features: pixel stats + histogram + FFT + block means.
    """
    from PIL import Image

    rows   = []
    images = (list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg"))
            + list(folder.glob("*.png"))  + list(folder.glob("*.JPG"))
            + list(folder.glob("*.JPEG")))
    images = images[:max_images]

    if not images:
        print(f"  ⚠  No images found in {folder}")
        return rows

    for p in tqdm(images, desc=f"    {'pos' if label else 'neg'} ({folder.name})",
                  leave=False):
        try:
            img  = Image.open(p).convert("L").resize((64, 64))
            arr  = np.array(img, dtype=np.float32) / 255.0

            stats = [arr.mean(), arr.std(), arr.min(), arr.max(),
                     np.percentile(arr, 10), np.percentile(arr, 25),
                     np.percentile(arr, 75), np.percentile(arr, 90)]

            hist, _ = np.histogram(arr, bins=64, range=(0, 1))
            hist     = hist.astype(np.float32) / (hist.sum() + 1e-9)

            fft = np.abs(np.fft.rfft2(arr)).flatten()[:200]

            blocks = [arr[i:i+16, j:j+16].mean()
                      for i in range(0, 64, 16)
                      for j in range(0, 64, 16)]

            feat = np.concatenate([stats, hist, fft, blocks])  # 288-dim
            rows.append((feat, label))
        except Exception:
            continue

    return rows

def train_image_model(name, pos_dirs, neg_dirs, max_per_class=1200):
    print(f"  Extracting features …")
    pos_rows, neg_rows = [], []

    for d in pos_dirs:
        pos_rows += extract_features_from_folder(Path(d), 1, max_per_class)
    for d in neg_dirs:
        neg_rows += extract_features_from_folder(Path(d), 0, max_per_class)

    if not pos_rows:
        print(f"  ⚠  No positive images. Skipping {name}."); return
    if not neg_rows:
        print(f"  ⚠  No negative images. Skipping {name}."); return

    # Balance classes
    n = min(len(pos_rows), len(neg_rows), max_per_class)
    np.random.shuffle(pos_rows)
    np.random.shuffle(neg_rows)
    all_rows = pos_rows[:n] + neg_rows[:n]
    np.random.shuffle(all_rows)

    X = np.array([r[0] for r in all_rows])
    y = np.array([r[1] for r in all_rows])
    feat_names = [f"feat_{i}" for i in range(X.shape[1])]

    print(f"  Dataset: {len(y)} images  |  pos={y.sum()}  neg={(y==0).sum()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("imp",   SimpleImputer()),
        ("sc",    StandardScaler()),
        ("model", make_model())
    ])
    pipe.fit(X_tr, y_tr)
    evaluate_and_save(name, pipe, X_te, y_te, feat_names)

# ─────────────────────────────────────────
# TABULAR DATASETS
# ─────────────────────────────────────────
def train_diabetes():
    print("\n=== 1/6  DIABETES ===")
    kaggle_download("uciml/pima-indians-diabetes-database", "diabetes")
    df = pd.read_csv(next((DATA_DIR / "diabetes").glob("*.csv")))
    train_tabular("diabetes", df, "Outcome")

def train_heart():
    print("\n=== 2/6  HEART DISEASE ===")
    kaggle_download("johnsmith88/heart-disease-dataset", "heart")
    df = pd.read_csv(next((DATA_DIR / "heart").glob("*.csv")))
    df["target"] = (df["target"] > 0).astype(int)
    train_tabular("heart", df, "target")

def train_kidney():
    print("\n=== 3/6  KIDNEY DISEASE ===")
    kaggle_download("mansoordaku/ckdisease", "kidney")
    df = pd.read_csv(next((DATA_DIR / "kidney").glob("*.csv")))
    df.replace({"?": np.nan, "\t?": np.nan}, inplace=True)
    df.columns = df.columns.str.strip()
    target = "classification" if "classification" in df.columns else df.columns[-1]
    df[target] = df[target].astype(str).str.strip().str.lower()
    df[target] = df[target].apply(lambda x: 1 if x.startswith("ckd") else 0)
    train_tabular("kidney", df, target)

# ─────────────────────────────────────────
# IMAGE DATASETS
# ─────────────────────────────────────────
def train_pneumonia():
    print("\n=== 4/6  PNEUMONIA (Chest X-Ray) ===")
    kaggle_download("paultimothymooney/chest-xray-pneumonia", "pneumonia")

    base    = DATA_DIR / "pneumonia"
    pos_dir = next((p for p in base.rglob("PNEUMONIA") if p.is_dir()), None)
    neg_dir = next((p for p in base.rglob("NORMAL")    if p.is_dir()), None)

    if not pos_dir or not neg_dir:
        print("  ⚠  PNEUMONIA/NORMAL folders not found.")
        print("  Folder contents:")
        for p in base.rglob("*"): print("   ", p.relative_to(base))
        return

    print(f"  PNEUMONIA: {len(list(pos_dir.glob('*.*')))} files")
    print(f"  NORMAL   : {len(list(neg_dir.glob('*.*')))} files")
    train_image_model("pneumonia", [pos_dir], [neg_dir])

def train_brain_tumor():
    print("\n=== 5/6  BRAIN TUMOR (MRI) ===")
    kaggle_download("masoudnickparvar/brain-tumor-mri-dataset", "brainTumor")

    base        = DATA_DIR / "brainTumor"
    notumor_dir = next((p for p in base.rglob("notumor") if p.is_dir()), None)

    if not notumor_dir:
        print("  ⚠  'notumor' folder not found.")
        print("  Folder contents:")
        for p in base.rglob("*"): print("   ", p.relative_to(base))
        return

    tumor_dirs = [p for p in notumor_dir.parent.iterdir()
                  if p.is_dir() and p.name != "notumor"]

    if not tumor_dirs:
        print("  ⚠  No tumor class folders found."); return

    print(f"  Tumor classes : {[d.name for d in tumor_dirs]}")
    print(f"  No-tumor      : {notumor_dir.name}")
    train_image_model("brainTumor", tumor_dirs, [notumor_dir])

def train_skin_cancer():
    print("\n=== 6/6  SKIN CANCER ===")

    kaggle_download("fanconic/skin-cancer-malignant-vs-benign", "skinCancer")

    base = DATA_DIR / "skinCancer"

    # Find folders dynamically
    pos_dirs = list(base.rglob("malignant"))
    neg_dirs = list(base.rglob("benign"))

    pos_dirs = [p for p in pos_dirs if p.is_dir()]
    neg_dirs = [p for p in neg_dirs if p.is_dir()]

    if not pos_dirs or not neg_dirs:
        print("  ⚠ malignant/benign folders not found.")
        for p in base.rglob("*"):
            print("   ", p.relative_to(base))
        return

    print(f"  Found {len(pos_dirs)} malignant dirs")
    print(f"  Found {len(neg_dirs)} benign dirs")

    total_pos = sum(len(list(d.glob("*.*"))) for d in pos_dirs)
    total_neg = sum(len(list(d.glob("*.*"))) for d in neg_dirs)

    print(f"  malignant: {total_pos} images")
    print(f"  benign   : {total_neg} images")

    train_image_model("skinCancer", pos_dirs, neg_dirs)

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║   MediScan AI — Full Training Pipeline   ║")
    print("╚══════════════════════════════════════════╝")

    check_kaggle()

    # Tabular models
    train_diabetes()
    train_heart()
    train_kidney()

    # Auto-install Pillow if missing
    try:
        from PIL import Image
    except ImportError:
        print("\n⚠  Pillow not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "-q"])

    # Image models
    train_pneumonia()
    train_brain_tumor()
    train_skin_cancer()

    # Save metrics (both filenames — app.py uses metrics.json)
    with open(MODEL_DIR / "all_metrics.json", "w") as f:
        json.dump(ALL_METRICS, f, indent=2)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(ALL_METRICS, f, indent=2)

    print("\n╔══════════════════════════════════════════╗")
    print("║            TRAINING COMPLETE             ║")
    print("╚══════════════════════════════════════════╝")
    for name, m in ALL_METRICS.items():
        print(f"  {name:<15}  Acc:{m['accuracy']}%   AUC:{m['auc']}%")

    print(f"\n✅ Models saved to: {MODEL_DIR}")
    print("\nNow restart the backend:")
    print("  cd backend && python app.py")
