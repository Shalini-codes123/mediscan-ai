#!/usr/bin/env bash
# MediScan AI — Local startup script
# Usage: chmod +x start.sh && ./start.sh

set -e
BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════╗"
echo "║         MediScan AI — Starting Up        ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Check Python ─────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}✗ Python 3 not found. Install from https://python.org${RESET}"
  exit 1
fi

# ── Check Node ───────────────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
  echo -e "${RED}✗ Node.js not found. Install from https://nodejs.org${RESET}"
  exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo -e "${YELLOW}→ Creating Python virtual environment...${RESET}"
  python3 -m venv venv
fi

echo -e "${YELLOW}→ Activating venv and installing backend deps...${RESET}"
source venv/bin/activate
pip install -r backend/requirements.txt -q

# ── Check if models exist ─────────────────────────────────────────────────────
MODEL_COUNT=$(ls backend/models/*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -lt 6 ]; then
  echo -e "${YELLOW}"
  echo "⚠  Models not found (found $MODEL_COUNT / 6)."
  echo "   Run this first to download datasets and train models:"
  echo -e "   ${BOLD}python scripts/download_and_train.py${RESET}${YELLOW}"
  echo ""
  echo "   (Requires Kaggle API key at ~/.kaggle/kaggle.json)"
  echo -e "${RESET}"
  read -p "Continue anyway? (models will fail to load) [y/N] " yn
  if [[ "$yn" != "y" && "$yn" != "Y" ]]; then
    exit 0
  fi
fi

# ── Install frontend deps ─────────────────────────────────────────────────────
if [ ! -d "frontend/node_modules" ]; then
  echo -e "${YELLOW}→ Installing frontend dependencies...${RESET}"
  cd frontend && npm install && cd ..
fi

# ── Start Flask backend ───────────────────────────────────────────────────────
echo -e "${GREEN}→ Starting Flask backend on http://localhost:5000${RESET}"
cd backend
python app.py &
FLASK_PID=$!
cd ..
sleep 2

# ── Start Vite frontend ───────────────────────────────────────────────────────
echo -e "${GREEN}→ Starting React frontend on http://localhost:3000${RESET}"
cd frontend
npm run dev &
VITE_PID=$!
cd ..

echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════╗"
echo "║   MediScan AI is running!                ║"
echo "║   Frontend → http://localhost:3000       ║"
echo "║   Backend  → http://localhost:5000       ║"
echo "║   Press Ctrl+C to stop both servers      ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
trap "echo 'Stopping...'; kill $FLASK_PID $VITE_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait