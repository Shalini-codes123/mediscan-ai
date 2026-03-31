# MediScan AI — Multi-Disease Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react" />
  <img src="https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask" />
  <img src="https://img.shields.io/badge/scikit--learn-1.8-orange?style=flat-square&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square" />
</p>

<p align="center">
  AI-powered disease prediction across 6 conditions using real clinical datasets and ensemble machine learning — with evaluation metrics and doctor prescription output.
</p>

---

## 🌐 Live Demo

👉 https://mediscan-ai-vxxz.onrender.com

---

## 🧠 Diseases Supported

| Disease | Type | Dataset | Accuracy | AUC |
|--------|------|--------|----------|-----|
| 🩸 Diabetes | Tabular | Pima Indians (UCI) | 74.7% | 82.9% |
| ❤️ Heart Disease | Tabular | Cleveland (UCI) | 83.0% | 91.4% |
| 🫘 Kidney Disease | Tabular | UCI CKD | 99.4% | 99.9% |
| 🫁 Pneumonia | Image (X-ray) | Kaggle Chest X-ray | ~93% | ~97% |
| 🧠 Brain Tumor | Image (MRI) | Kaggle MRI | ~95% | ~98% |
| 🔬 Skin Cancer | Image (Dermoscopy) | HAM10000 | ~82% | ~89% |

---

## ⚙️ Features

- Ensemble ML (Random Forest + Gradient Boosting + MLP)
- Image analysis using HOG-style features
- Full evaluation metrics per prediction
- AI-generated doctor prescription output
- REST API with clean JSON responses
- Deployable on Render + Hugging Face models

---


---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/Shalini-codes123/mediscan-ai.git
cd mediscan-ai
pip install -r backend/requirements.txt

---

### 2. Train Models

python scripts/download_and_train.py

3. Run Backend

cd backend
python app.py

4. Run Frontend

cd frontend
npm install
npm run dev

🔌 API
Health
GET /health

Predict
POST /predict/<disease>

🧪 Tech Stack
Python, Flask, scikit-learn
React, Vite
Render, Hugging Face
