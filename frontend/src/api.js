/**
 * MediScan API client
 * In dev:  Vite proxies /predict → http://localhost:5000/predict
 * In prod: served from same origin (Flask serves built React static files)
 */

const BASE = ""; // same origin — Vite proxy handles dev routing

export async function predict(diseaseId, mode, formData, imageFile) {
  let body;

  if (mode === "image") {
    if (!imageFile) throw new Error("Image file required");
    const b64 = await toBase64(imageFile);
    body = JSON.stringify({ image_b64: b64 });
  } else {
    body = JSON.stringify(formData);
  }

  const res = await fetch(`${BASE}/predict/${diseaseId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${res.status}`);
  }
  return res.json();
}

export async function fetchAllMetrics() {
  const res = await fetch(`${BASE}/metrics`);
  if (!res.ok) throw new Error("Failed to load metrics");
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

function toBase64(file) {
  return new Promise((res, rej) => {
    const reader = new FileReader();
    reader.onload = () => res(reader.result.split(",")[1]);
    reader.onerror = rej;
    reader.readAsDataURL(file);
  });
}
