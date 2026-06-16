# 🌿 EcoGPT — Multi-Agent RAG Environmental Advisory System

**AMD–TCS Hackathon** | Location-aware environmental intelligence for any point on Earth

---

## What it does

EcoGPT accepts any city name or `lat, lon` coordinate and runs a full environmental analysis pipeline:

```
Location → Live data fetch (Open-Meteo CAMS/ERA5)
         → 8 Specialist Agents (AQI · Pollution · Plantation · Water · Urban · Carbon · Energy · Synthesis)
         → Hybrid RAG (ChromaDB dense + BM25 sparse → Reciprocal Rank Fusion)
         → Structured report + satellite map + impact charts + what-if simulator
```

All values are **live and location-specific** — a different result for Paris vs Jaisalmer vs Nairobi. Every estimated figure is explicitly marked *(estimated)*; every assumption is listed.

---

## Quick start

```bash
git clone https://github.com/<YOUR_USERNAME>/ecogpt.git
cd ecogpt
```

Open `ecogpt_hackathon.ipynb` → **Run All Cells**.  
Cell 1 auto-installs dependencies and prints a capability report. All agents work with zero external services (deterministic fallbacks for everything).

---

## LLM backends (auto-detected, priority order)

### 1. AMD vLLM — Qwen3-30B-A3B on AMD Instinct (recommended for hackathon)

Start the vLLM server in a terminal on the AMD Jupyter platform:

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-30B-A3B \
    --served-model-name Qwen3-30B-A3B \
    --api-key abc-123 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
```

Monitor GPU utilisation: `watch rocm-smi`

Then in the notebook call the helper:
```python
setup_amd_vllm()      # prints the launch command and probes the server
LLM = LLMBackend()    # reinit after server is ready
```

Or set env vars manually:
```bash
export AMD_VLLM_URL="http://localhost:8000"
export AMD_VLLM_MODEL="Qwen3-30B-A3B"
export AMD_VLLM_KEY="abc-123"
```

### 2. Ollama (local, AMD ROCm-accelerated)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve & ollama pull mistral:7b-instruct
```

### 3. HuggingFace Inference API (cloud fallback)

```python
import os; os.environ["HF_TOKEN"] = "hf_xxx"
```

### 4. Deterministic engine (no LLM)

Every agent has a full rule-based scientific core — the pipeline runs completely without any LLM. LLMs only polish the narrative text.

---

## LLM telemetry

Every inference call is logged automatically:

```python
llm_usage_summary()   # full per-call table: backend, model, latency, tokens in/out
```

The Streamlit sidebar shows a live telemetry panel. The dashboard footer shows cumulative token counts and average latency after each run.

---

## Production UI (Streamlit)

```bash
pip install streamlit
streamlit run ecogpt_app.py
```

On AMD's remote Jupyter platform, reach it via `/proxy/8501/` or an SSH tunnel.

Features: AQI + temperature + rainfall metric cards, satellite map (Esri + MODIS NDVI) with OSM plantable plots and rivers, modern impact charts, solar/wind energy plan, RAG Q&A chat tab, LLM telemetry sidebar.

---

## Data sources

| Source | What it provides | Key |
|--------|-----------------|-----|
| Open-Meteo CAMS | **Current** PM2.5, PM10, CO, NO2 (model analysis step) | None |
| Open-Meteo ERA5 | **Current** temperature, humidity, apparent temp, wind | None |
| Open-Meteo ERA5 | Today's solar radiation → kWh/m²/day | None |
| OSM Overpass API | Plantable land, water bodies, rivers (polygons + relations) | None |
| IoT sensor CSV | Real ground-truth readings (auto-merged from `enviro_sensorvalues_*.csv`) | — |
| Climate-zone model | Offline fallback when all APIs unreachable | — |

Temperature and AQI are **current readings** (not multi-day averages) — the live data fetch uses Open-Meteo's `current` object, not hourly means.

---

## Notebook cell map

| Cell | File | Purpose |
|------|------|---------|
| 1 | `01_setup.py` | Package install, capability detection, GPU check |
| 2 | `02_data.py` | Sensor CSV loader + synthetic calibrated dataset |
| 2b | `02b_kaggle.py` | Kaggle city-AQI baseline cross-check |
| 3 | `03_eda.py` | Exploratory data analysis plots |
| 4 | `04_kb.py` | Knowledge base — 60+ scientific chunks |
| 5 | `05_rag.py` | Hybrid RAG: ChromaDB + BM25 + RRF |
| 6 | `06_tools.py` | AQI engine, geocoding, OSM, live data fetch |
| 7 | `07_llm.py` | **AMD vLLM / Ollama / HF backend + telemetry** |
| 8 | `08_agents.py` | 8 specialist agents (all deterministic + optional LLM polish) |
| 9 | `09_orchestrator.py` | Pipeline orchestrator + `ask_ecogpt()` entry point |
| 10 | `10_chat.py` | Interactive ipywidgets chat UI |
| 11 | `11_dashboard.py` | Satellite map + 8-panel modern matplotlib dashboard |
| 12 | `12_demos.py` | Demo: Kolkata / Sundarbans / London |
| 13 | `13_whatif.py` | What-if intervention sliders |
| 13b | `13b_streamlit.py` | Dual-mode: Streamlit app + full in-notebook fallback UI |

---

## Key design decisions

- **No hallucination of figures**: every number comes from a formula, a live API, or the knowledge base. Estimates are always flagged.
- **OSM water/land accuracy**: both polygon (`waterway=riverbank`) and relation (multipolygon) features fetched; geometry limit 400 nodes per element to handle large parks and rivers.
- **`simulate_intervention` is location-aware**: `urban_area_ha` derived from the actual survey radius (`π·r²·100 ha`), not hardcoded to KMC's 20,500 ha.
- **Live data is current**: `fetch_live_environment` uses Open-Meteo's `current` parameter — a single real-time reading, not a 7-day average.

---

## Files

```
ecogpt_hackathon.ipynb   Main notebook (auto-assembled from build/)
ecogpt_core.py           Engine module (auto-assembled — do not edit by hand)
ecogpt_app.py            Streamlit UI
build/                   Source cells (edit these; run assemble.py to regenerate)
enviro_sensorvalues_*.csv  Drop real sensor data here to auto-merge
```
