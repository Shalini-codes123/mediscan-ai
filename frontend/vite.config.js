import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // All /predict, /metrics, /health calls proxy to Flask
      "/predict": { target: "http://localhost:5000", changeOrigin: true },
      "/metrics": { target: "http://localhost:5000", changeOrigin: true },
      "/health": { target: "http://localhost:5000", changeOrigin: true },
    },
  },
  build: {
    outDir: "../backend/static",
    emptyOutDir: true,
  },
});
