import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "backend" / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def check_kaggle():
    from kaggle.api.kaggle_api_extended import KaggleApi

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle key missing")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()
    print("✓ Kaggle ready")

def kaggle_download(dataset, path):
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest = DATA_DIR / path
    dest.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {dataset}...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(dest), unzip=True)
    print(f"  ✓ Done")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer

ALL_METRICS = {}

def make_model():
    return VotingClassifier([
        ("rf", RandomForestClassifier(n_estimators=200)),
        ("gb", GradientBoostingClassifier()),
        ("mlp", MLPClassifier(max_iter=300))
    ], voting="soft")


def evaluate_and_save(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"[{name}] Acc:{round(acc*100,2)}% AUC:{round(auc*100,2)}%")

    joblib.dump(pipe, MODEL_DIR / f"{name}.pkl")
    ALL_METRICS[name] = {"accuracy": acc, "auc": auc}


# ─────────────────────────────────────────
# SAFE TABULAR TRAINING
# ─────────────────────────────────────────
def train_tabular(name, df, target):

    if target not in df.columns:
        print("⚠ Target missing")
        return

    print(f"  Initial shape: {df.shape}")

    y = df[target]
    X_df = df.drop(columns=[target])

    # Encode categorical
    for col in X_df.columns:
        if X_df[col].dtype == object:
            X_df[col] = X_df[col].astype(str).str.strip()
            X_df[col] = LabelEncoder().fit_transform(X_df[col])

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median())

    df_clean = X_df.copy()
    df_clean[target] = y

    df_clean = df_clean[df_clean[target].notna()]

    print(f"  After cleaning: {df_clean.shape}")

    if len(df_clean) < 20:
        print("⚠ Too small dataset. Skipping.")
        return

    X = df_clean.drop(columns=[target]).values
    y = df_clean[target].values

    if len(np.unique(y)) < 2:
        print("⚠ Only one class. Skipping.")
        return

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    pipe = Pipeline([
        ("imp", SimpleImputer()),
        ("sc", StandardScaler()),
        ("model", make_model())
    ])

    pipe.fit(X_tr, y_tr)
    evaluate_and_save(name, pipe, X_te, y_te)


# ─────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────
def train_diabetes():
    print("\n=== DIABETES ===")
    kaggle_download("uciml/pima-indians-diabetes-database", "diabetes")

    df = pd.read_csv(next((DATA_DIR/"diabetes").glob("*.csv")))
    train_tabular("diabetes", df, "Outcome")


def train_heart():
    print("\n=== HEART ===")
    kaggle_download("johnsmith88/heart-disease-dataset", "heart")

    df = pd.read_csv(next((DATA_DIR/"heart").glob("*.csv")))
    df["target"] = (df["target"] > 0).astype(int)

    train_tabular("heart", df, "target")


def train_kidney():
    print("\n=== KIDNEY ===")
    kaggle_download("mansoordaku/ckdisease", "kidney")

    df = pd.read_csv(next((DATA_DIR/"kidney").glob("*.csv")))

    print("  Raw shape:", df.shape)

    df.replace({"?": np.nan, "\t?": np.nan}, inplace=True)
    df.columns = df.columns.str.strip()

    target = "classification" if "classification" in df.columns else df.columns[-1]

    df[target] = df[target].astype(str).str.strip().str.lower()

    print("  Unique target:", df[target].unique())

    # FIXED LOGIC
    df[target] = df[target].apply(
        lambda x: 1 if x.startswith("ckd") else 0
    )

    train_tabular("kidney", df, target)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("MediScan AI — Stable Pipeline")

    check_kaggle()

    train_diabetes()
    train_heart()
    train_kidney()

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(ALL_METRICS, f, indent=2)

    print("\n✅ ALL DONE")