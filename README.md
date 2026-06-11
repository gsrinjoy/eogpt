# 🌿 EcoGPT — Multi-Agent RAG Environmental Advisory System
AMD–TCS Hackathon | Location-aware environmental intelligence from IoT sensor streams

Single self-contained notebook: **`ecogpt_hackathon.ipynb`** (8 agents, hybrid RAG, AQI engine, what-if simulator, chat UI).

## Run on the AMD Jupyter platform

```bash
# In a Jupyter Terminal (or prefix with ! in a notebook cell):
git clone https://github.com/<YOUR_USERNAME>/ecogpt.git
cd ecogpt
```

Open `ecogpt_hackathon.ipynb` → **Run All Cells**. Cell 1 auto-installs dependencies and prints a capability report (fallbacks activate for anything unavailable).

**Optional LLM backend** (pipeline is fully functional without one):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve & ollama pull mistral:7b-instruct   # AMD ROCm-accelerated
# or: os.environ["HF_TOKEN"]="hf_xxx" in a cell before Cell 7
```

## 🚀 Production UI (Streamlit)

```bash
pip install streamlit
streamlit run ecogpt_app.py        # or just run the notebook's last code cell
```

`ecogpt_app.py` (polished UI: metric cards, satellite map with rivers & plantable plots, impact
charts, solar plan, RAG Q&A chat) imports `ecogpt_core.py` (the engine — auto-assembled from the
same notebook cells). On a remote AMD Jupyter instance, reach the app via the platform proxy
(`/proxy/8501/`) or an SSH tunnel.

## Usage
- **Cell 10** — chat widget: enter a city or `lat, lon` → full report; follow-up questions are RAG-grounded with citations
- **Cell 11** — `build_dashboard(lat, lon)`: pollution heatmap + plantation zones + CO₂ projection
- **Cell 12** — demo scenarios (Kolkata dense-urban, Sundarbans riverbank, London)
- **Cell 13** — what-if intervention sliders
- Drop any `enviro_sensorvalues_*.csv` (same schema) beside the notebook → auto-merged on Cell 2 re-run
