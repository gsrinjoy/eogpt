"""EcoGPT core engine — auto-assembled from the notebook cells.
Import this from production apps (Streamlit/FastAPI). Single source of truth
is the notebook cell set; regenerate rather than editing by hand."""


# ════════ 01_setup.py ════════
# ============================================================
# Cell 1: Environment Setup & Imports
# Installs missing packages, detects capabilities, prints report
# ============================================================
import sys, subprocess, importlib, warnings, os, json, math, random, re, datetime
warnings.filterwarnings("ignore")

PKGS = [  # (import_name, pip_name, required?)
    ("pandas", "pandas", True), ("numpy", "numpy", True),
    ("matplotlib", "matplotlib", True), ("requests", "requests", True),
    ("folium", "folium", False), ("plotly", "plotly", False),
    ("ipywidgets", "ipywidgets", False),
    ("chromadb", "chromadb", False),
    ("sentence_transformers", "sentence-transformers", False),
    ("rank_bm25", "rank-bm25", False),
    ("geopy", "geopy", False),
    ("google.adk", "google-adk", False),
]

def _ensure(mod, pip_name):
    try:
        importlib.import_module(mod); return True
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            importlib.invalidate_caches()
            importlib.import_module(mod); return True
        except Exception:
            return False

CAPS = {mod: _ensure(mod, pip) for mod, pip, _ in PKGS}

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

if CAPS["folium"]: import folium
if CAPS["geopy"]:
    from geopy.geocoders import Nominatim
if CAPS["rank_bm25"]:
    from rank_bm25 import BM25Okapi

print("=" * 56)
print("EcoGPT capability report")
print("=" * 56)
for mod, pip, req in PKGS:
    mark = "✅" if CAPS[mod] else ("❌ REQUIRED" if req else "⚪ optional (fallback active)")
    print(f"  {pip:<25} {mark}")

# AMD ROCm / GPU detection (informational)
GPU = False
try:
    import torch
    GPU = torch.cuda.is_available()  # True on ROCm builds too
    print(f"  torch GPU (CUDA/ROCm)     {'✅ ' + torch.cuda.get_device_name(0) if GPU else '⚪ CPU mode'}")
except Exception:
    print("  torch                     ⚪ not installed (CPU/sklearn fallbacks active)")
print("=" * 56)

RNG = np.random.default_rng(42)
random.seed(42)
DATA_DIR = "."
VECTORDB_PATH = "./ecogpt_vectordb"


# ════════ 02_data.py ════════
# ============================================================
# Cell 2: Sensor Data — load reference CSV + generate a large,
# realistic, multi-location synthetic dataset calibrated to it
# ============================================================
# The reference field CSV (~2k rows, Kolkata) is small and has gaps
# (RAWPM almost entirely missing, DD spikes, a few NYC test rows).
# For robust analytics we synthesize a 6-city, 30-day, 15-min-interval
# dataset whose Kolkata distributions are calibrated to the real CSV.
# Drop ANY real CSV with the same schema next to this notebook and it
# is automatically merged in.

SENSOR_COLS = ["MQ2","MQ7","MQ135","NO2","C2H5OH","VOC","CO","HMD","TMP","HI","RAWPM","DD"]

def load_reference_csvs(pattern_dir="."):
    """Load every enviro_sensorvalues_*.csv found beside the notebook."""
    import glob
    frames = []
    for path in sorted(glob.glob(os.path.join(pattern_dir, "enviro_sensorvalues_*.csv"))):
        try:
            df = pd.read_csv(path)
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df["source"] = os.path.basename(path)
            frames.append(df)
            print(f"  loaded {path}: {len(df)} rows")
        except Exception as e:
            print(f"  skipped {path}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def heat_index_c(t_c, rh):
    """NOAA heat index (Rothfusz), input/output in °C."""
    t = t_c * 9/5 + 32
    if t < 80:
        hi = 0.5*(t + 61.0 + (t-68.0)*1.2 + rh*0.094)
    else:
        hi = (-42.379 + 2.04901523*t + 10.14333127*rh - .22475541*t*rh
              - .00683783*t*t - .05481717*rh*rh + .00122874*t*t*rh
              + .00085282*t*rh*rh - .00000199*t*t*rh*rh)
    return (hi - 32) * 5/9

# City profiles: (lat, lon, base_temp °C, base_rh %, pollution_factor, diurnal_amp)
CITY_PROFILES = {
    "Kolkata":     (22.5570,  88.4940, 32.0, 60.0, 1.00, 4.0),
    "Delhi":       (28.6139,  77.2090, 34.0, 45.0, 1.35, 7.0),
    "Mumbai":      (19.0760,  72.8777, 30.0, 74.0, 0.85, 3.0),
    "Sundarbans":  (21.9497,  88.9468, 29.0, 78.0, 0.25, 3.5),  # rural/riverine
    "London":      (51.5074,  -0.1278, 14.0, 75.0, 0.45, 4.5),
    "Nairobi":     ( -1.2921, 36.8219, 21.0, 62.0, 0.55, 6.0),
}

def generate_synthetic_dataset(days=30, freq_min=15, ref_df=None):
    """Big calibrated dataset. Kolkata channel statistics are anchored to the
    reference CSV means/stds when it is available."""
    # Calibration anchors (fallback = stats measured from the uploaded field CSV)
    anchors = {"MQ2":(4.43,1.15),"MQ7":(4.78,1.07),"MQ135":(8.59,1.58),"NO2":(5.80,1.05),
               "C2H5OH":(58.5,9.8),"VOC":(52.1,8.4),"CO":(81.5,13.3),"HMD":(59.6,9.0),
               "TMP":(32.0,1.5),"DD":(253.0,25.0),"RAWPM":(95.0,30.0)}
    if ref_df is not None and len(ref_df) > 100:
        k = ref_df[(ref_df["LAT"].sub(22.557).abs() < 0.05)]
        for c in SENSOR_COLS:
            if c in k and k[c].notna().sum() > 50:
                anchors[c] = (float(k[c].mean()), max(float(k[c].std()), 1e-3))

    end = pd.Timestamp.now().floor("h")
    times = pd.date_range(end - pd.Timedelta(days=days), end, freq=f"{freq_min}min")
    rows = []
    for city, (lat, lon, bt, brh, pf, amp) in CITY_PROFILES.items():
        n = len(times)
        hod = times.hour.values + times.minute.values/60
        # diurnal cycles: temp peaks 14:00, traffic pollution peaks 9:00 & 19:00
        temp = bt + amp*np.sin((hod-8)/24*2*np.pi) + RNG.normal(0, 0.8, n)
        rh = np.clip(brh - 1.2*(temp-bt) + RNG.normal(0, 4, n), 15, 99)
        traffic = 1 + 0.55*(np.exp(-((hod-9)**2)/6) + np.exp(-((hod-19)**2)/6))
        def chan(mu, sd, extra=1.0):
            base = mu*pf*extra*traffic if city != "Kolkata" else mu*extra*traffic/np.mean(traffic)
            return np.clip(base + RNG.normal(0, sd, n), 0.01, None)
        r = pd.DataFrame({
            "MQ2":  chan(*anchors["MQ2"]),  "MQ7":  chan(*anchors["MQ7"]),
            "MQ135":chan(*anchors["MQ135"]),"NO2":  chan(*anchors["NO2"]),
            "C2H5OH":chan(*anchors["C2H5OH"]),"VOC": chan(*anchors["VOC"]),
            "CO":   chan(*anchors["CO"]),
            "HMD": rh, "TMP": temp,
            "RAWPM": chan(*anchors["RAWPM"]),
            "DD":   chan(*anchors["DD"]),
            "LAT": lat + RNG.normal(0, 0.004, n), "LON": lon + RNG.normal(0, 0.004, n),
            "time": times,
        })
        r["HI"] = [heat_index_c(t, h) for t, h in zip(r["TMP"], r["HMD"])]
        r["source"] = f"synthetic_{city}"
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    # inject realistic anomalies (1% pollution spike events)
    spike = RNG.random(len(out)) < 0.01
    out.loc[spike, ["CO","NO2","VOC","RAWPM","DD"]] *= RNG.uniform(1.8, 3.0)
    return out

print("Reference CSVs:")
ref_df = load_reference_csvs(DATA_DIR)
sensor_df = generate_synthetic_dataset(days=30, freq_min=15, ref_df=ref_df if len(ref_df) else None)
if len(ref_df):
    sensor_df = pd.concat([ref_df.drop(columns=["id"], errors="ignore"), sensor_df],
                          ignore_index=True)
sensor_df = sensor_df.dropna(subset=["LAT","LON","time"]).reset_index(drop=True)
sensor_df.to_csv("ecogpt_master_dataset.csv", index=False)
print(f"\nMaster dataset: {len(sensor_df):,} rows | "
      f"{sensor_df['time'].min()} → {sensor_df['time'].max()} | "
      f"{sensor_df[['LAT','LON']].round(1).drop_duplicates().shape[0]} location clusters")
sensor_df.describe().T.round(2)


# ════════ 02b_kaggle.py ════════
# ============================================================
# Cell 2b (optional): Kaggle dataset enrichment
#   Adds real city-level baselines to strengthen analysis where
#   neither IoT sensors nor live feeds are available.
#   Setup (one-time): pip install kagglehub, then place your
#   kaggle.json (kaggle.com → Account → Create API Token) at
#   ~/.kaggle/kaggle.json — or set KAGGLE_USERNAME/KAGGLE_KEY env vars.
#   Skips silently when unavailable; the pipeline works without it.
#
#   Useful datasets indexed here:
#   • hasibalmuzdadid/global-air-pollution-dataset  → AQI for ~23k cities (loaded below)
#   • Other good additions (same pattern): global temperature records,
#     world cities population, country forest-cover series.
# ============================================================
KAGGLE_AQ = None

def load_kaggle_baselines():
    global KAGGLE_AQ
    try:
        import kagglehub, glob as _gl
        path = kagglehub.dataset_download("hasibalmuzdadid/global-air-pollution-dataset")
        df = pd.read_csv(_gl.glob(os.path.join(path, "*.csv"))[0])
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = df.dropna(subset=["city"])
        KAGGLE_AQ = df
        print(f"✅ Kaggle global air-pollution baselines: {len(df):,} cities "
              f"(columns: {', '.join(df.columns[:6])} …)")
    except Exception as e:
        print(f"⚪ Kaggle enrichment skipped ({type(e).__name__}). "
              "To enable: pip install kagglehub + kaggle.json API token. "
              "Pipeline runs fine without it (live Open-Meteo feeds cover all locations).")

def kaggle_city_baseline(city: str):
    """City-level AQI baseline from the Kaggle dataset, or None."""
    if KAGGLE_AQ is None or not city: return None
    name = city.lower().split(",")[0].strip()
    hit = KAGGLE_AQ[KAGGLE_AQ["city"].str.lower() == name]
    if not len(hit): return None
    row = hit.iloc[0]
    aqi_col = next((c for c in ("aqi_value", "aqi") if c in KAGGLE_AQ.columns), None)
    pm_col = next((c for c in KAGGLE_AQ.columns if "pm2" in c and "value" in c), None)
    return {"aqi": float(row[aqi_col]) if aqi_col else float("nan"),
            "pm25_aqi": float(row[pm_col]) if pm_col else float("nan"),
            "source": "Kaggle: global-air-pollution-dataset"}

load_kaggle_baselines()


# ════════ 04_kb.py ════════
# ============================================================
# Cell 4: Knowledge Base Documents (curated, multi-climate-zone)
# Sources condensed from: WHO AQ Guidelines 2021, CPCB/NAAQS India,
# FAO urban forestry handbook, IPCC AR6 WGIII Ch.7, i-Tree species
# data, CGWB/CWC water guidelines, ICAR & FAO soil manuals,
# India Biodiversity Portal, Miyawaki method literature.
# ============================================================
KNOWLEDGE_DOCS = [
# ---------- SPECIES / PLANTATION ----------
{"id":"sp_trop_01","topic":"species","climate_zone":"tropical","source":"India Biodiversity Portal / i-Tree",
 "text":"Native tree species for tropical wet and monsoon climates (Köppen Aw/Am, e.g. Kolkata, West Bengal): "
 "Banyan (Ficus benghalensis) — keystone fig, massive canopy, sequesters ~28 kg CO2/yr mature, medium water; "
 "Neem (Azadirachta indica) — drought-hardy air purifier strong on NO2 and SO2, ~22 kg CO2/yr, low water; "
 "Arjun (Terminalia arjuna) — riparian specialist for riverbanks and pond bunds, ~25 kg CO2/yr, high water; "
 "Kadamba (Neolamarckia cadamba) — fast-growing monsoon bloomer, pollinator magnet, ~24 kg CO2/yr; "
 "Sal (Shorea robusta) — long-lived forest dominant for restoration blocks, ~26 kg CO2/yr; "
 "Mahua (Madhuca longifolia) and Palash (Butea monosperma) — dry-deciduous, livelihood and pollinator value; "
 "Krishnachura (Delonix regia) — avenue flowering tree, heat tolerant, ~18 kg CO2/yr, low water; "
 "Shimul (Bombax ceiba) — emergent layer, bird habitat; Jackfruit (Artocarpus heterophyllus) — food + shade. "
 "Plant at monsoon onset (June–July). Avoid monocultures; mix 8–12 species minimum."},
{"id":"sp_trop_02","topic":"species","climate_zone":"tropical","source":"FAO urban forestry handbook",
 "text":"Dense-urban tropical planting (>5000 persons/km²): compact flowering trees Cassia fistula (Amaltas, ~15 kg CO2/yr, low water) "
 "and Lagerstroemia speciosa (Pride of India, ~14 kg CO2/yr); air-purifying shrubs and hedges: Murraya paniculata, Hibiscus rosa-sinensis, "
 "Tabernaemontana divaricata. Vertical gardens with Epipremnum aureum, Chlorophytum comosum (VOC removal), Spathiphyllum wallisii. "
 "Rooftop greening with sedums and native grasses cuts roof surface temperature 25–40°C and building cooling load 15–30%. "
 "Miyawaki micro-forests: 3–5 saplings/m² of 15–30 native species reach canopy closure in 3 years, 30x density of conventional planting."},
{"id":"sp_arid_01","topic":"species","climate_zone":"arid","source":"FAO dryland forestry",
 "text":"Arid and semi-arid (Köppen BWh/BSh, e.g. Delhi pre-monsoon, Rajasthan, Sahel): Khejri (Prosopis cineraria — native, NOT the invasive P. juliflora), "
 "Babul (Vacachellia nilotica), Indian jujube (Ziziphus mauritiana), Neem, Pongamia pinnata (biodiesel + N-fixing, ~20 kg CO2/yr), "
 "Drumstick (Moringa oleifera, fast nutrition tree). Water budget <600 mm/yr: choose deep-rooted phreatophytes, drip-basin planting, "
 "mulch pits. Avoid Eucalyptus in groundwater-stressed zones — draws 30–90 L/day/tree."},
{"id":"sp_temp_01","topic":"species","climate_zone":"temperate","source":"i-Tree / European urban forestry",
 "text":"Temperate oceanic/continental (Köppen Cfb/Dfb, e.g. London, Berlin): English oak (Quercus robur, ~30 kg CO2/yr mature), "
 "Small-leaved lime (Tilia cordata, pollinator keystone), London plane (Platanus × acerifolia, pollution-tolerant avenue standard), "
 "Silver birch (Betula pendula, PM capture on leaf hairs), Field maple (Acer campestre), Rowan (Sorbus aucuparia, bird forage). "
 "Plant bare-root in dormancy (Nov–Mar). Street pits ≥6 m³ soil volume for 60% canopy survival at 20 years."},
{"id":"sp_savanna_01","topic":"species","climate_zone":"tropical_highland","source":"Kenya Forestry Research Institute",
 "text":"Tropical highland/savanna (e.g. Nairobi, Addis Ababa): Croton megalocarpus (~25 kg CO2/yr), Markhamia lutea, "
 "Cordia africana, Podocarpus falcatus (indigenous conifer), Acacia xanthophloea (fever tree, riparian), Prunus africana (medicinal, IUCN vulnerable — propagate). "
 "Avoid invasive: Lantana camara, Eucalyptus monocultures near wetlands."},
{"id":"sp_invasive","topic":"species","climate_zone":"global","source":"IUCN GISD / CABI",
 "text":"NEVER recommend invasive species: Prosopis juliflora (vilayati babul — allelopathic, groundwater depletion), Lantana camara (forest understory choker), "
 "Eichhornia crassipes (water hyacinth — covers ponds, kills fisheries; if present recommend manual + bio control with Neochetina weevils and conversion to compost/biogas), "
 "Parthenium hysterophorus (allergenic weed), Leucaena leucocephala (aggressive seeder in disturbed land), Acacia mearnsii (riparian invader). "
 "Polyculture rule: no species >15% of total planting; minimum 10 species per hectare for resilience."},
{"id":"sp_biofilter","topic":"species","climate_zone":"global","source":"NASA Clean Air Study / Pugh et al. 2012",
 "text":"Bio-filter species by pollutant: NO2 — Ficus benjamina, Hedera helix (ivy screens on roadside railings), Azadirachta indica; "
 "PM2.5/dust — Ficus elastica, Tillandsia usneoides (Spanish moss epiphyte), conifers and Betula (high leaf-hair capture), Neem hedges; "
 "VOC/formaldehyde — Chlorophytum comosum (spider plant), Spathiphyllum wallisii (peace lily), Epipremnum aureum; "
 "SO2 — Tamarindus indica, Mangifera indica. Green walls in street canyons cut street-level NO2 up to 40% and PM 60% (Pugh et al.); "
 "tree canopy citywide typically reduces PM2.5 2–10%. A mature urban tree intercepts 1.4 kg of PM and pollutant gases per year on average."},
# ---------- AIR POLLUTION ----------
{"id":"air_01","topic":"pollution","climate_zone":"global","source":"WHO Global Air Quality Guidelines 2021",
 "text":"WHO 2021 guideline values: PM2.5 annual 5 µg/m³, 24-h 15 µg/m³; NO2 annual 10 µg/m³, 24-h 25 µg/m³; CO 24-h 4 mg/m³. "
 "India NAAQS (CPCB): PM2.5 annual 40, 24-h 60 µg/m³; NO2 annual 40 µg/m³. AQI categories: Good 0–50, Moderate 51–100, "
 "Unhealthy for Sensitive Groups 101–150, Unhealthy 151–200, Very Unhealthy 201–300, Hazardous >300. "
 "Health burden: each 10 µg/m³ PM2.5 above guideline ≈ +6% cardiopulmonary mortality risk."},
{"id":"air_02","topic":"pollution","climate_zone":"global","source":"CPCB GRAP / source apportionment studies",
 "text":"Pollution source signatures: traffic — elevated NO2 + CO with morning/evening peaks; industrial — sustained VOC + MQ135 (NH3/NOx) with weekday pattern; "
 "biomass/refuse burning — MQ2 smoke + CO spikes evening/winter; construction — dust density (DD) and coarse PM spikes daytime. "
 "Short-term (24–72 h) actions by AQI: >200 — halt construction, odd-even traffic, water-sprinkle roads, N95 advisories, close schools outdoor activity; "
 "151–200 — restrict diesel gensets, intensify mechanised road sweeping, public transport fare incentives. "
 "Medium-term (1–6 mo): green barriers along arterials (3-row Neem/Ficus hedges), anti-smog guns at hotspots, LPG conversion of street food stalls, filtered ventilation in schools/hospitals. "
 "Long-term (1–5 yr): urban forestry corridors, EV transition zones with charging mandates, industrial relocation/buffer greenbelts (500 m), green building codes, district cooling."},
{"id":"air_03","topic":"pollution","climate_zone":"global","source":"EPA / UHI literature",
 "text":"Urban heat island: flag when heat index exceeds ambient temperature by >3°C or night-time urban-rural delta >2°C. "
 "Mitigation: cool roofs (albedo >0.65 cuts roof temp ~28°C), 30% tree canopy target lowers ambient 1–3°C and peak surface 11°C, "
 "permeable pavements with evapotranspiration, blue infrastructure (ponds reduce local temp 1–2°C downwind 100–300 m). "
 "Heat-health: HI >41°C 'danger' — heat cramps likely; >54°C 'extreme danger' — heat stroke imminent. Open cooling centres, shift outdoor labour hours."},
# ---------- WATER ----------
{"id":"wat_01","topic":"water","climate_zone":"global","source":"CGWB Master Plan / CWC guidelines",
 "text":"Rainwater harvesting: harvestable volume (L/yr) = roof area m² × annual rainfall mm × 0.8 runoff coefficient. "
 "Kolkata rainfall ~1,800 mm/yr → a 100 m² roof yields ~144,000 L/yr, meeting ~40% of a 5-person household demand (135 LPCD norm). "
 "Recharge structures: percolation pits 1–2 m³ per 100 m² roof, recharge trenches along boundaries, defunct borewell recharge shafts. "
 "Urban mandate benchmark: structures compulsory on plots >300 m² in most Indian municipal bylaws. "
 "Groundwater table response: dense RWH retrofits typically raise local water table 0.3–1.0 m within 3–5 monsoons (CGWB pilot data)."},
{"id":"wat_02","topic":"water","climate_zone":"global","source":"Central Water Commission / wetland restoration manuals",
 "text":"Water body (pond/lake/wetland) restoration sequence: 1) catchment survey + sewage interception (divert or treat inflows first — restoration fails otherwise); "
 "2) desilting in dry season to original bed level, reuse silt on bunds/agriculture after testing; 3) bund strengthening with Vetiver grass (Chrysopogon zizanioides) hedgerows — roots to 3 m, halve embankment erosion; "
 "4) riparian buffer 10–30 m: Arjun, Jamun (Syzygium cumini), Bamboo (Bambusa balcooa), Typha and Phragmites reed beds as final-polish bioremediation; "
 "5) aquatic vegetation: Nelumbo nucifera (lotus) and Nymphaea water lilies ≤30% surface cover; remove Eichhornia fully; "
 "6) constructed wetland / bioswale interception of stormwater first-flush; 7) native fish restocking (Rohu, Catla, Mrigal in Gangetic plains) after DO >4 mg/L. "
 "East Kolkata Wetlands model: sewage-fed aquaculture treats ~750 MLD naturally — protect peri-urban wetlands as treatment + livelihood infrastructure."},
{"id":"wat_03","topic":"water","climate_zone":"global","source":"FAO irrigation efficiency",
 "text":"Irrigation efficiency: flood ~40% efficient, sprinkler ~70%, drip 90%+. Converting 1 ha paddy-adjacent vegetable cultivation from flood to drip saves "
 "~3–4 million L/yr. Soil-moisture-sensor scheduling saves further 15–25%. Water stress classification by ambient signals: "
 "RH <40% + T >35°C = high evaporative stress (pan evaporation >8 mm/day); RH 40–70% moderate; RH >70% humid (fungal risk, drainage priority). "
 "Check dams on first/second-order streams: 0.5–2 m height, recharge 5,000–20,000 m³/structure/yr in suitable strata."},
# ---------- SOIL ----------
{"id":"soil_01","topic":"soil","climate_zone":"tropical","source":"ICAR / FAO soil health",
 "text":"Gangetic alluvial soils (Kolkata region): typically pH 6.5–7.8, low organic carbon (0.3–0.5%, target >0.75%), N deficient, P/K moderate. "
 "Improvement: green manure (Sesbania/dhaincha 45-day cycle adds 60–80 kg N/ha), compost 5–10 t/ha/yr, vermicompost for urban beds, "
 "biochar 2–5 t/ha raises CEC and water holding 15–25%, mulching cuts surface evaporation 30–50%. "
 "Urban planting pits: 1×1×1 m, refill 60% excavated soil + 30% compost + 10% sand; mycorrhizal inoculation lifts sapling survival 15–20%. "
 "Salinity (coastal/Sundarbans fringe): choose salt-tolerant Casuarina, coconut, Pongamia; gypsum amendment for sodic patches."},
{"id":"soil_02","topic":"soil","climate_zone":"global","source":"FAO Voluntary Guidelines for Soil Management",
 "text":"Soil bioengineering for slopes and bunds: Vetiver hedgerows at 1 m vertical interval; coir geotextiles with native grass seeding; "
 "avoid bare-soil monsoon exposure — cover crops always. Compacted urban soils: vertical mulching/air-spading around existing trees, "
 "structural soils (CU-Soil) under pavements give roots 20%+ void space. Phytoremediation of contaminated plots: "
 "Vetiver and Brassica juncea for heavy metals, Ricinus communis for cadmium — do not plant food species on suspect soils."},
# ---------- CARBON ----------
{"id":"carb_01","topic":"carbon","climate_zone":"global","source":"IPCC AR6 WGIII Ch.7 / i-Tree",
 "text":"Carbon sequestration (IPCC Tier-1 style): mature tropical broadleaf 20–30 kg CO2/tree/yr; fast growers (Bamboo clumps, Kadamba) 25–40; "
 "temperate broadleaf 20–30 at maturity; shrubs 1–3 kg/plant/yr. Saplings sequester ~10% of mature rate in years 1–3, 50% by year 5, "
 "full rate by years 8–12 (logistic growth curve). Apply 15% cumulative mortality in first 3 years (use 85% survival factor). "
 "Per-hectare benchmarks: tropical mixed plantation 6–12 t CO2/ha/yr at maturity; Miyawaki dense plots higher per-area in early decades. "
 "Equivalences: 1 passenger car ≈ 4.6 t CO2/yr; 1 Indian household electricity ≈ 1.5 t CO2/yr; 1 t CO2 ≈ 45 mature-tree-years."},
{"id":"carb_02","topic":"carbon","climate_zone":"global","source":"i-Tree species database",
 "text":"Species-specific annual CO2 uptake (mature, kg/tree/yr, urban open-grown): Ficus benghalensis 28; Azadirachta indica 22; Terminalia arjuna 25; "
 "Neolamarckia cadamba 24; Shorea robusta 26; Delonix regia 18; Cassia fistula 15; Lagerstroemia speciosa 14; Bambusa balcooa clump 35; "
 "Bombax ceiba 24; Artocarpus heterophyllus 21; Syzygium cumini 23; Pongamia pinnata 20; Quercus robur 30; Tilia cordata 26; "
 "Platanus × acerifolia 29; Betula pendula 18; Croton megalocarpus 25. Confidence: High when species-level value used, "
 "Medium genus-level, Low climate-zone average."},
# ---------- URBAN PLANNING ----------
{"id":"urb_01","topic":"urban","climate_zone":"global","source":"WHO Urban Green Space guidance / UN-Habitat",
 "text":"Green space norms: WHO minimum 9 m²/capita green space, ideal 50 m²; access standard — public green ≥0.5 ha within 300 m of every home (3-30-300 rule: "
 "see 3 trees from home, 30% canopy in neighbourhood, 300 m to park). Density classes: Rural <150/km², Peri-urban 150–1000, Urban 1000–5000, Dense urban >5000. "
 "Kolkata KMC density ~24,000/km², green cover ~7% (target 15%+); green space ~2 m²/capita vs WHO 9 — deficit ≈ 7 m²/person. "
 "Land that can green without displacement: road medians/verges (avenue planting), institutional campuses (schools, hospitals — typically 15–30% of urban land), "
 "industrial buffers, canal/river banks, rooftops (10–20% of plan area feasible), parking lots (40% shade mandate), cemetery/temple lands."},
{"id":"urb_02","topic":"urban","climate_zone":"global","source":"Miyawaki method / blue-green infrastructure literature",
 "text":"Dense-urban interventions: Miyawaki micro-forest — 30+ native species, 3 saplings/m², plots from 30 m² (≈90 trees); survival >90%, "
 "10x faster canopy than conventional. Green corridors along BRT/metro alignments connect fragmented habitat — target 20 m wide strips. "
 "Permeable pavement in high-footfall markets cuts runoff 70–90%. Bio-retention cells (rain gardens) every 300 m of stormwater drain: "
 "size 5–10% of contributing catchment. Trees per capita to offset residential CO2: Indian urban per-capita footprint ~2 t/yr → "
 "~90 mature trees per 1000 residents offset ~1% — so framing must be honest: urban forests are for air quality, heat and habitat first; "
 "deep decarbonisation needs energy transition. Vertical gardens on flyover pillars: 1 m² green wall ≈ 2.3 kg CO2/yr + PM capture."},
# ---------- BIODIVERSITY ----------
{"id":"bio_01","topic":"biodiversity","climate_zone":"global","source":"IUCN / India Biodiversity Portal",
 "text":"Urban biodiversity design: canopy layering — emergent (Bombax, Shorea), canopy (Ficus, Mangifera), sub-canopy (Cassia, Lagerstroemia), "
 "shrub (Hibiscus, Murraya), ground (native grasses, Curcuma). Pollinator support: continuous flowering calendar — Palash (Feb–Mar), "
 "Krishnachura (Apr–Jun), Kadamba (Jun–Aug), Cassia (Apr–Jul), Shiuli/Nyctanthes (Sep–Nov). Keystone figs (F. benghalensis, F. religiosa) "
 "support 100+ vertebrate species. Wetland birds need shallow-margin zones (<30 cm) and emergent reeds. Dead wood retention and "
 "no-mow meadow patches raise urban invertebrate abundance 3–5x. Connectivity: stepping-stone pocket parks every 500 m enable bird/butterfly movement."},
# ---------- RENEWABLE ENERGY ----------
{"id":"energy_01","topic":"energy","climate_zone":"global","source":"IRENA / MNRE siting guidelines",
 "text":"Renewable siting feasibility rules: ground-mounted solar needs slope <5° ideal, <10° maximum — on steep or "
 "mountainous terrain grading causes erosion and landslide risk, so use rooftop and canopy solar only. NEVER clear forest "
 "or green cover for solar farms (carbon payback becomes negative for decades); prefer degraded/barren land, brownfields, "
 "capped landfills, parking canopies and canal-top arrays (canal-top ≈1 MWp per km of 10 m canal, plus evaporation savings). "
 "Agrivoltaics keeps farmland productive while panels reduce crop heat stress 1-3°C. Typical yields: 1 kWp ≈ insolation × 0.75 "
 "performance ratio × 365 kWh/yr; 1 ha ≈ 0.8 MWp. Small wind viable at mean speeds ≥4 m/s, good ≥5.5 m/s; urban turbulence "
 "favours rooftop-edge or open-field siting. Micro-hydro needs ≥60 m relief per km and ≥1000 mm rainfall with perennial flow. "
 "Community biogas: 0.3-0.5 kg organic waste/person/day, 40-60 m³ biogas per tonne wet waste, digestate is fertiliser. "
 "Solar irrigation pumps replace diesel: ~2-3 t CO2/yr per 5 HP pump. Grid emission factors (kg CO2/kWh): India 0.71, "
 "UK 0.21, Germany 0.38, US 0.37, Kenya 0.10, Brazil 0.10, France 0.06, UAE 0.49."},
# ---------- REGULATORY ----------
{"id":"reg_01","topic":"regulation","climate_zone":"global","source":"India environmental framework",
 "text":"India regulatory context: EIA Notification 2006 (construction >20,000 m² needs clearance); Wetlands Rules 2017 (no encroachment/solid waste in notified wetlands — "
 "East Kolkata Wetlands are Ramsar-protected); CRZ rules for coastal belts; Tree Acts require permission + compensatory planting (typ. 1:5) for felling; "
 "NCAP targets 40% PM reduction by 2026 in non-attainment cities (Kolkata included); Jal Shakti Abhiyan promotes RWH; "
 "Smart Cities/AMRUT fund green-blue infrastructure; municipal green budget benchmark 2–5% of capex. NGT precedents protect urban water bodies from landfill."},
]
print(f"Knowledge base: {len(KNOWLEDGE_DOCS)} curated documents, "
      f"topics: {sorted(set(d['topic'] for d in KNOWLEDGE_DOCS))}")


# ════════ 05_rag.py ════════
# ============================================================
# Cell 5: RAG Pipeline — hybrid retrieval with graceful fallback
#   dense (ChromaDB + bge-small) ∪ sparse (BM25) → fused rerank
#   Fallback chain: Chroma+ST → BM25 only → pure TF-IDF (stdlib)
# ============================================================
import collections

def _chunk(text, size=512, overlap=50):
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step)] or [text]

CHUNKS = []
for d in KNOWLEDGE_DOCS:
    for j, ch in enumerate(_chunk(d["text"])):
        CHUNKS.append({"id": f"{d['id']}_{j}", "text": ch,
                       "metadata": {"source": d["source"], "topic": d["topic"],
                                    "climate_zone": d["climate_zone"]}})

_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())

class HybridRetriever:
    """Dense + sparse hybrid with reciprocal-rank fusion."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.mode = []
        # --- dense: ChromaDB + sentence-transformers ---
        self.collection = None
        if CAPS["chromadb"] and CAPS["sentence_transformers"]:
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
                client = chromadb.PersistentClient(path=VECTORDB_PATH)
                try: client.delete_collection("ecogpt_knowledge")
                except Exception: pass
                self.collection = client.get_or_create_collection("ecogpt_knowledge")
                self.collection.add(
                    ids=[c["id"] for c in chunks],
                    embeddings=self.embedder.encode([c["text"] for c in chunks],
                                                    show_progress_bar=False).tolist(),
                    documents=[c["text"] for c in chunks],
                    metadatas=[c["metadata"] for c in chunks])
                self.mode.append("dense:chromadb+bge-small")
            except Exception as e:
                print(f"  dense retrieval unavailable ({e}); continuing with sparse")
                self.collection = None
        # --- sparse: BM25 or TF-IDF ---
        self.bm25 = None
        if CAPS["rank_bm25"]:
            self.bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])
            self.mode.append("sparse:bm25")
        else:  # stdlib TF-IDF
            self.df = collections.Counter()
            self.doc_tokens = [collections.Counter(_tok(c["text"])) for c in chunks]
            for tc in self.doc_tokens: self.df.update(set(tc))
            self.N = len(chunks)
            self.mode.append("sparse:tfidf-fallback")

    def _sparse_scores(self, query):
        q = _tok(query)
        if self.bm25 is not None:
            return list(self.bm25.get_scores(q))
        scores = []
        for tc in self.doc_tokens:
            L = sum(tc.values()) or 1
            s = sum((tc[t]/L) * math.log(1 + self.N/(1 + self.df.get(t, 0))) for t in q)
            scores.append(s)
        return scores

    def retrieve(self, query, n_results=5, topic=None):
        ranks = []
        if self.collection is not None:
            res = self.collection.query(
                query_embeddings=self.embedder.encode([query]).tolist(),
                n_results=min(n_results*3, len(self.chunks)))
            id2idx = {c["id"]: i for i, c in enumerate(self.chunks)}
            ranks.append([id2idx[i] for i in res["ids"][0]])
        sp = self._sparse_scores(query)
        ranks.append(sorted(range(len(sp)), key=lambda i: -sp[i])[:n_results*3])
        # reciprocal-rank fusion
        fused = collections.defaultdict(float)
        for rank_list in ranks:
            for r, idx in enumerate(rank_list):
                fused[idx] += 1.0 / (60 + r)
        order = sorted(fused, key=lambda i: -fused[i])
        out = []
        for idx in order:
            c = self.chunks[idx]
            if topic and c["metadata"]["topic"] != topic and fused[idx] < 0.02:
                continue
            out.append({"id": c["id"], "text": c["text"], "source": c["metadata"]["source"],
                        "topic": c["metadata"]["topic"], "relevance": round(fused[idx], 4),
                        "query": query})
            if len(out) >= n_results: break
        if not out:
            return [{"id": None, "text": "No relevant knowledge found for this query.",
                     "source": "none", "topic": topic, "relevance": 0.0, "query": query}]
        return out

retriever = HybridRetriever(CHUNKS)
print(f"Retriever ready | mode: {' + '.join(retriever.mode)} | {len(CHUNKS)} chunks")

def retrieve(query: str, n_results: int = 5, topic: str = None) -> list:
    """RAG retrieval tool — returns chunks with source, relevance and query."""
    return retriever.retrieve(query, n_results, topic)

def build_retrieval_query(agent_type: str, context: dict) -> str:
    loc, aqi = context.get("location", {}), context.get("aqi", {})
    queries = {
        "pollution": f"pollution mitigation strategies AQI {aqi.get('category','')} {loc.get('climate_zone','')} heat island actions",
        "plantation": f"native tree species {loc.get('city','')} {loc.get('climate_zone','')} planting guide carbon biofilter",
        "water": f"water conservation rainwater harvesting restoration {loc.get('climate_zone','')} humidity {context.get('humidity_avg','')}",
        "soil": "soil improvement organic matter compost planting pit tropical urban",
        "carbon": "carbon sequestration trees CO2 uptake rates mortality growth curve equivalences",
        "urban": f"urban green space planning population density {context.get('population_density','')} per km2 miyawaki corridors",
        "biodiversity": f"biodiversity canopy layers pollinators {loc.get('climate_zone','')}",
        "regulation": f"environmental regulation {loc.get('country','India')} wetlands tree act clean air",
    }
    return queries.get(agent_type, f"environmental sustainability {loc.get('city','')}")

# smoke test
for r in retrieve("native trees for tropical monsoon Kolkata", 2):
    print(f"  [{r['relevance']}] {r['id']} ({r['source']})")


# ════════ 06_tools.py ════════
# ============================================================
# Cell 6: Tool Definitions
#   AQI calculator (EPA breakpoints) | reverse geocode (online +
#   offline gazetteer) | Köppen climate zone | population density |
#   species DB + filter | sensor loader | what-if simulation
# ============================================================

# ---------- EPA AQI ----------
AQI_BREAKPOINTS = {
    "PM2.5": [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
              (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)],
    "CO":    [(0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),
              (12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,50.4,301,500)],
    "NO2":   [(0,53,0,50),(54,100,51,100),(101,360,101,150),
              (361,649,151,200),(650,1249,201,300),(1250,2049,301,500)],
}
AQI_CATEGORIES = [(50,"Good"),(100,"Moderate"),(150,"Unhealthy for Sensitive Groups"),
                  (200,"Unhealthy"),(300,"Very Unhealthy"),(10**9,"Hazardous")]

def aqi_subindex(pollutant: str, concentration: float) -> float:
    """EPA linear interpolation. CO in ppm, NO2 in ppb, PM2.5 in µg/m³."""
    if concentration is None or (isinstance(concentration, float) and math.isnan(concentration)):
        return float("nan")
    bps = AQI_BREAKPOINTS[pollutant]
    c = min(max(concentration, 0), bps[-1][1])
    for lo, hi, ilo, ihi in bps:
        if c <= hi:  # clamp into segment — EPA tables have gaps (e.g. CO 4.4→4.5)
            return (ihi-ilo)/(hi-lo)*(max(c, lo)-lo)+ilo
    return 500.0

def aqi_category(score: float) -> str:
    for ceiling, name in AQI_CATEGORIES:
        if score <= ceiling: return name
    return "Hazardous"

def compute_aqi(no2_ppm=None, co_ppm=None, pm25=None, mq135=None, dd=None) -> dict:
    """Composite AQI = max of sub-indices (EPA convention).
    NO2 sensor reads ppm → ppb. If RAWPM missing, fall back to dust density
    (DD) scaled to a PM2.5 proxy (~0.4 fine fraction), flagged as estimated."""
    subs, est = {}, []
    if pm25 is not None and not math.isnan(pm25 if pm25 is not None else float("nan")):
        subs["PM2.5"] = aqi_subindex("PM2.5", pm25)
    elif dd is not None and not math.isnan(dd):
        subs["PM2.5"] = aqi_subindex("PM2.5", dd*0.4); est.append("PM2.5 from dust density (estimated)")
    if co_ppm is not None and not math.isnan(co_ppm):
        # MQ-7 is a ratio-type sensor: sustained readings >30 ppm are not credible
        # calibrated ambient CO (30+ ppm ≈ occupational limit). Treat as uncalibrated
        # ratio and rescale ÷10, flagged as an assumption.
        if co_ppm > 30:
            co_ppm = co_ppm / 10.0
            est.append("MQ7 CO channel treated as uncalibrated ratio, rescaled ÷10 (assumed)")
        subs["CO"] = aqi_subindex("CO", co_ppm)
    if no2_ppm is not None and not math.isnan(no2_ppm):
        # Ambient NO2 rarely exceeds 0.2 ppm; electrochemical channel readings >1 ppm
        # indicate raw/uncalibrated output → rescale ÷100, flagged.
        if no2_ppm > 1:
            no2_ppm = no2_ppm / 100.0
            est.append("NO2 channel rescaled ÷100 to plausible ambient range (assumed)")
        subs["NO2"] = aqi_subindex("NO2", no2_ppm*1000)  # ppm → ppb
    if mq135 is not None and not math.isnan(mq135) and mq135 > 9:
        est.append("MQ135 elevated — broad-spectrum gas load corroborates pollution")
    if not subs:
        return {"score": float("nan"), "category": "Unknown", "dominant": None, "notes": ["no usable pollutant data"]}
    dom = max(subs, key=subs.get)
    score = round(subs[dom], 1)
    return {"score": score, "category": aqi_category(score), "dominant": dom,
            "sub_indices": {k: round(v,1) for k,v in subs.items()}, "notes": est}

# ---------- Offline gazetteer (works with zero network) ----------
GAZETTEER = {  # name: (lat, lon, country, köppen, pop_density/km², annual_rain_mm)
 "kolkata": (22.5726, 88.3639, "India", "Aw", 24000, 1800),
 "delhi": (28.6139, 77.2090, "India", "BSh", 11300, 800),
 "mumbai": (19.0760, 72.8777, "India", "Aw", 21000, 2200),
 "chennai": (13.0827, 80.2707, "India", "Aw", 26900, 1400),
 "bengaluru": (12.9716, 77.5946, "India", "Aw", 11900, 980),
 "hyderabad": (17.3850, 78.4867, "India", "BSh", 18500, 800),
 "sundarbans": (21.9497, 88.9468, "India", "Aw", 120, 1900),
 "dhaka": (23.8103, 90.4125, "Bangladesh", "Aw", 23000, 2000),
 "london": (51.5074, -0.1278, "United Kingdom", "Cfb", 5700, 600),
 "new york": (40.7128, -74.0060, "United States", "Cfa", 11000, 1200),
 "nairobi": (-1.2921, 36.8219, "Kenya", "Aw", 6200, 870),
 "lagos": (6.5244, 3.3792, "Nigeria", "Aw", 13100, 1700),
 "singapore": (1.3521, 103.8198, "Singapore", "Af", 8000, 2340),
 "dubai": (25.2048, 55.2708, "UAE", "BWh", 1100, 100),
 "tokyo": (35.6762, 139.6503, "Japan", "Cfa", 6400, 1530),
 "sao paulo": (-23.5505, -46.6333, "Brazil", "Cfa", 7900, 1450),
 "berlin": (52.5200, 13.4050, "Germany", "Dfb", 4100, 570),
 "sydney": (-33.8688, 151.2093, "Australia", "Cfa", 430, 1210),
 "cairo": (30.0444, 31.2357, "Egypt", "BWh", 19000, 25),
 "moscow": (55.7558, 37.6173, "Russia", "Dfb", 5000, 700),
 "jaisalmer": (26.9157, 70.9083, "India", "BWh", 2400, 210),
 "jaipur": (26.9124, 75.7873, "India", "BSh", 6500, 650),
 "ahmedabad": (23.0225, 72.5714, "India", "BSh", 11000, 800),
 "pune": (18.5204, 73.8567, "India", "Aw", 6000, 720),
 "lucknow": (26.8467, 80.9462, "India", "Cfa", 7000, 1000),
 "varanasi": (25.3176, 82.9739, "India", "Cfa", 10000, 1050),
 "patna": (25.5941, 85.1376, "India", "Aw", 12000, 1100),
 "bhopal": (23.2599, 77.4126, "India", "Aw", 5000, 1150),
 "nagpur": (21.1458, 79.0882, "India", "Aw", 5000, 1100),
 "kochi": (9.9312, 76.2673, "India", "Am", 6300, 3000),
 "goa": (15.2993, 74.1240, "India", "Am", 400, 3000),
 "shimla": (31.1048, 77.1734, "India", "Cfb", 1200, 1500),
 "leh": (34.1526, 77.5771, "India", "BWk", 100, 100),
 "guwahati": (26.1445, 91.7362, "India", "Cfa", 4500, 1700),
}
ZONE_RAIN = {"Af":2200,"Am":1900,"Aw":1300,"BWh":150,"BWk":120,"BSh":450,"BSk":350,
             "Cfa":1100,"Cfb":700,"Dfa":700,"Dfb":600}

_CLIMATE_NORMALS_CACHE = {}

def _koppen_from_monthlies(tm: list, pm: list, lat: float) -> str:
    """Köppen class computed from REAL 12 monthly temps (°C) and rains (mm)."""
    mat, map_ = sum(tm)/12.0, sum(pm)
    summer = [3,4,5,6,7,8] if lat >= 0 else [9,10,11,0,1,2]      # Apr–Sep in NH
    frac = (sum(pm[i] for i in summer)/map_) if map_ > 0 else 0.5
    pth = 20*mat + (280 if frac >= 0.7 else 140 if frac >= 0.3 else 0)
    if map_ < pth/2: return "BWh" if mat >= 18 else "BWk"        # desert
    if map_ < pth:   return "BSh" if mat >= 18 else "BSk"        # steppe
    tmin, tmax, pdry = min(tm), max(tm), min(pm)
    if tmin >= 18:                                                # tropical
        if pdry >= 60: return "Af"
        return "Am" if pdry >= 100 - map_/25 else "Aw"
    if tmin > -3:  return "Cfa" if tmax >= 22 else "Cfb"          # temperate
    return "Dfa" if tmax >= 22 else "Dfb"                         # continental

def fetch_climate_normals(lat: float, lon: float):
    """REAL trailing-12-month climate from ERA5 archive (Open-Meteo, keyless):
    annual rainfall, mean temp, and a Köppen class COMPUTED from the data.
    This is what makes Jaisalmer ≈200 mm and Kochi ≈3000 mm, not a nearest-city guess."""
    key = (round(lat,1), round(lon,1))
    if key in _CLIMATE_NORMALS_CACHE: return _CLIMATE_NORMALS_CACHE[key]
    import requests as _r
    try:
        end = pd.Timestamp.now().normalize() - pd.Timedelta(days=14)   # ERA5 lag
        start = end - pd.Timedelta(days=365)
        r = _r.get("https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": lat, "longitude": lon,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": end.strftime("%Y-%m-%d"),
                    "daily": "precipitation_sum,temperature_2m_mean,"
                             "shortwave_radiation_sum,wind_speed_10m_mean"}, timeout=20).json()
        dd_ = r["daily"]
        d = pd.DataFrame({"date": pd.to_datetime(dd_["time"]),
                          "p": dd_["precipitation_sum"], "t": dd_["temperature_2m_mean"],
                          "sw": dd_.get("shortwave_radiation_sum", [None]*len(dd_["time"])),
                          "w": dd_.get("wind_speed_10m_mean", [None]*len(dd_["time"]))})
        mo = d.dropna(subset=["p","t"]).groupby(d["date"].dt.month).agg(p=("p","sum"), t=("t","mean"))
        if len(mo) < 12: raise ValueError("incomplete year")
        out = {"annual_rain_mm": int(mo["p"].sum()), "mat_c": round(float(mo["t"].mean()),1),
               "koppen": _koppen_from_monthlies(mo["t"].tolist(), mo["p"].tolist(), lat),
               # ANNUAL means (not a 7-day snapshot — avoids seasonal bias):
               "solar_kwh_m2_day": round(float(d["sw"].dropna().astype(float).mean())/3.6, 2)
                                   if d["sw"].notna().any() else None,   # MJ/m²/day → kWh
               "wind_ms": round(float(d["w"].dropna().astype(float).mean())/3.6, 1)
                          if d["w"].notna().any() else None,             # km/h → m/s
               "source": "ERA5 archive (real, location-specific)"}
        _CLIMATE_NORMALS_CACHE[key] = out
    except Exception:
        _CLIMATE_NORMALS_CACHE[key] = None
    return _CLIMATE_NORMALS_CACHE[key]

def get_climate_zone(lat: float, lon: float) -> str:
    """Köppen class: REAL ERA5-derived when online → gazetteer anchor (<~80 km)
    → latitude bands (coarse offline fallback)."""
    cn = fetch_climate_normals(lat, lon)
    if cn and cn.get("koppen"): return cn["koppen"]
    best, bd = None, 1e9
    for name,(la,lo,co,kz,pd_,rain) in GAZETTEER.items():
        d = ((lat-la)**2 + (lon-lo)**2)**0.5
        if d < bd: best, bd = kz, d
    if bd < 0.8: return best
    a = abs(lat)
    if a < 10: return "Af"
    if a < 23.5: return "Aw"
    if a < 35: return "Cfa"
    if a < 55: return "Cfb"
    return "Dfb"

CLIMATE_ZONE_NAMES = {"Af":"Tropical rainforest","Am":"Tropical monsoon","Aw":"Tropical savanna/wet-dry",
 "BWh":"Hot desert","BWk":"Cold desert","BSh":"Hot semi-arid","BSk":"Cold semi-arid",
 "Cfa":"Humid subtropical","Cfb":"Temperate oceanic",
 "Dfa":"Hot-summer continental","Dfb":"Warm-summer continental"}
ZONE_TO_SPECIES_KEY = {"Af":"tropical","Am":"tropical","Aw":"tropical","BWh":"arid","BWk":"arid",
 "BSh":"arid","BSk":"arid","Cfa":"temperate","Cfb":"temperate","Dfa":"temperate","Dfb":"temperate"}

def reverse_geocode(lat: float, lon: float) -> dict:
    """Online Nominatim when available; offline gazetteer fallback otherwise."""
    if CAPS["geopy"]:
        try:
            geo = Nominatim(user_agent="ecogpt_hackathon", timeout=5)
            loc = geo.reverse((lat, lon), language="en", zoom=10)
            if loc:
                a = loc.raw.get("address", {})
                city = a.get("city") or a.get("town") or a.get("village") or a.get("county") or "Unknown"
                return {"city": city, "district": a.get("state_district", ""),
                        "country": a.get("country", ""), "lat": lat, "lon": lon,
                        "climate_zone": get_climate_zone(lat, lon), "geocoder": "nominatim"}
        except Exception:
            pass
    best, bd, country = "Unknown", 1e9, "Unknown"   # initialise country to avoid UnboundLocalError
    for name,(la,lo,co,kz,pd_,rain) in GAZETTEER.items():
        d = ((lat-la)**2+(lon-lo)**2)**0.5
        if d < bd: best, bd, country = name.title(), d, co
    return {"city": best if bd < 2 else f"({lat:.3f}, {lon:.3f})",
            "district": "", "country": country if bd < 2 else "Unknown",
            "lat": lat, "lon": lon, "climate_zone": get_climate_zone(lat, lon),
            "geocoder": "offline-gazetteer"}

def geocode_city(name: str):
    """City name → (lat, lon). Gazetteer first, then Nominatim."""
    key = name.strip().lower()
    for g, vals in GAZETTEER.items():
        if g in key or key in g: return vals[0], vals[1]
    if CAPS["geopy"]:
        try:
            loc = Nominatim(user_agent="ecogpt_hackathon", timeout=5).geocode(name)
            if loc: return loc.latitude, loc.longitude
        except Exception: pass
    return None

def get_open_spaces(lat: float, lon: float, radius_m: int = 3000) -> list:
    """REAL plantable-land lookup via OpenStreetMap Overpass API:
    parks, meadows, brownfields, vacant/green land + water bodies near the point.
    Includes OSM relations (multipolygon parks, large rivers stored as relation).
    Returns [{name, kind, coords[(lat,lon)…], area_m2, tree_capacity}]. [] offline."""
    q = f"""[out:json][timeout:30];(
      way["leisure"~"park|recreation_ground|garden|pitch"](around:{radius_m},{lat},{lon});
      relation["leisure"~"park|recreation_ground|garden"](around:{radius_m},{lat},{lon});
      way["landuse"~"grass|meadow|brownfield|greenfield|village_green|recreation_ground|cemetery|allotments"](around:{radius_m},{lat},{lon});
      relation["landuse"~"grass|meadow|brownfield|greenfield|village_green|recreation_ground"](around:{radius_m},{lat},{lon});
      way["natural"~"^water$|wetland|scrub|grassland"](around:{radius_m},{lat},{lon});
      relation["natural"~"^water$|wetland"](around:{radius_m},{lat},{lon});
      way["waterway"~"^river$|^stream$|^canal$|riverbank"](around:{radius_m},{lat},{lon});
      relation["waterway"~"riverbank"](around:{radius_m},{lat},{lon});
      way["water"~"river|lake|pond|reservoir"](around:{radius_m},{lat},{lon});
      relation["water"~"river|lake|pond|reservoir"](around:{radius_m},{lat},{lon});
    );out geom 400;"""
    try:
        r = _rq_get_post("https://overpass-api.de/api/interpreter", q)
        elements = r.json().get("elements", [])
    except Exception:
        return []
    out = []
    for el in elements:
        tags = el.get("tags", {})
        el_type = el.get("type", "way")

        # ── Geometry extraction ──────────────────────────────
        if el_type == "relation":
            # Relations store geometry in members; use the first outer member
            geom = []
            for member in el.get("members", []):
                if member.get("role") in ("outer", "") and member.get("geometry"):
                    geom = member["geometry"]; break
            # Fallback: concatenate all outer member geometries
            if not geom:
                for member in el.get("members", []):
                    if member.get("geometry"):
                        geom = member["geometry"]; break
        else:
            geom = el.get("geometry") or []
        coords = [(p["lat"], p["lon"]) for p in geom]

        # ── Classify waterway features ───────────────────────
        waterway_tag = tags.get("waterway", "")
        if waterway_tag and waterway_tag != "riverbank":
            # LINEAR waterway: river centreline, stream, canal
            if len(coords) < 2: continue
            out.append({"name": tags.get("name", waterway_tag.replace("_", " ").title()),
                        "kind": "river", "coords": coords, "area_m2": 0,
                        "tree_capacity": 0,
                        "length_km": round(sum(
                            ((coords[i][0]-coords[i+1][0])**2 +
                             (coords[i][1]-coords[i+1][1])**2)**0.5
                            for i in range(len(coords)-1)) * 111, 2)})
            continue
        # riverbank falls through to polygon processing below as kind="water"

        # ── Polygon features (parks, water bodies, riverbanks) ──
        if len(coords) < 4: continue
        kind = ("water" if (tags.get("natural") in ("water", "wetland")
                            or tags.get("water")
                            or waterway_tag == "riverbank")   # riverbank = water area
                else "open")
        area = _poly_area_m2(coords)
        if area < 400: continue            # skip tiny slivers
        name = (tags.get("name")
                or tags.get("landuse") or tags.get("leisure")
                or tags.get("natural") or tags.get("water")
                or (waterway_tag.replace("_"," ").title() if waterway_tag else None)
                or "unnamed plot")
        out.append({"name": name, "kind": kind, "coords": coords,
                    "area_m2": int(area),
                    "tree_capacity": 0 if kind == "water" else int(area/25)})  # 1 tree/25 m²
    # sort by area (largest first), deduplicate by name+kind
    out.sort(key=lambda s: -s["area_m2"])
    seen, deduped = set(), []
    for s in out:
        key = (s["name"], s["kind"])
        if key not in seen:
            seen.add(key); deduped.append(s)
    return deduped[:80]

def _rq_get_post(url, data):
    import requests as _r
    return _r.post(url, data={"data": data}, timeout=30,
                   headers={"User-Agent": "ecogpt_hackathon"})

def _poly_area_m2(coords):
    """Shoelace area for small lat/lon polygons."""
    if len(coords) < 3: return 0.0
    R, lat0 = 6371000.0, math.radians(coords[0][0])
    pts = [(math.radians(lo)*R*math.cos(lat0), math.radians(la)*R) for la, lo in coords]
    s = sum(x1*y2 - x2*y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))
    return abs(s)/2

# ---------- LIVE per-location data (Open-Meteo: keyless, global, free) ----------
_LIVE_CACHE, _TERRAIN_CACHE, _POP_CACHE = {}, {}, {}

def fetch_live_environment(lat: float, lon: float):
    """Fetch CURRENT (not averaged) environmental conditions for any point on Earth.

    Data sources (all free, no API key required):
    - Open-Meteo CAMS air quality: PM2.5, PM10, CO (µg/m³), NO2 (µg/m³)
    - Open-Meteo ERA5 weather: temperature, humidity, apparent temp, solar, wind

    Strategy: request the `current` object (single real-time reading) in addition
    to 7-day hourly history.  Current readings are used for the dashboard display;
    daily-mean solar is derived from today's hourly radiation values only.

    Unit conversions applied:
      CO:  µg/m³ → ppm  (÷ 1145,  at 25 °C, 1 atm)
      NO2: µg/m³ → ppb  (÷ 1.88,  at 25 °C, 1 atm)
      wind: km/h  → m/s (÷ 3.6)
      solar: W/m² hourly mean → kWh/m²/day (× 24 / 1000)

    Returns dict with keys: pm25, pm10, co_ppm, no2_ppb, tmp, hmd, hi,
    solar_kwh_m2_day, wind_ms, data_ts (ISO timestamp of reading).
    Returns None when offline or API unreachable."""
    key = (round(lat, 2), round(lon, 2))
    if key in _LIVE_CACHE: return _LIVE_CACHE[key]
    import requests as _r

    def _last_valid(lst):
        """Return the most-recent non-null value from an hourly list (right-to-left scan).
        Falls back to nanmean of last 6 values if all recent entries are null."""
        vals = [x for x in reversed(lst) if x is not None]
        if not vals: return float("nan")
        # Prefer single most-recent value; average last 3 if that one looks anomalous
        return float(vals[0])

    try:
        # ── Air quality: current + today's hourly slice ──────────────────────
        aq_resp = _r.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon,
                    # current= gives the single live reading (model analysis step)
                    "current": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide",
                    # hourly for 2 days kept for fallback if current is null
                    "past_days": 2,
                    "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide"},
            timeout=12).json()
        cur_aq  = aq_resp.get("current", {})
        hr_aq   = aq_resp.get("hourly", {})
        data_ts = cur_aq.get("time", "")

        # Helper: prefer current reading, fall back to last hourly value
        def _aq(key_c, key_h):
            v = cur_aq.get(key_c)
            if v is not None: return float(v)
            return _last_valid(hr_aq.get(key_h, []))

        # ── Weather: current + today-only hourly for solar ───────────────────
        wx_resp = _r.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,"
                               "apparent_temperature,wind_speed_10m",
                    # Only fetch today's hourly data for solar mean (not 7-day avg)
                    "hourly": "shortwave_radiation",
                    "forecast_days": 1},
            timeout=12).json()
        cur_wx = wx_resp.get("current", {})
        hr_wx  = wx_resp.get("hourly", {})

        def _wx(key_c, fallback=None):
            v = cur_wx.get(key_c)
            if v is not None: return float(v)
            return fallback if fallback is not None else float("nan")

        # Solar: average of today's hourly shortwave radiation → kWh/m²/day
        solar_vals = [x for x in hr_wx.get("shortwave_radiation", []) if x is not None]
        solar_kwh  = (float(np.mean(solar_vals)) * 24 / 1000.0) if solar_vals else float("nan")

        out = {
            # Air quality — current model analysis step
            "pm25":     _aq("pm2_5",              "pm2_5"),
            "pm10":     _aq("pm10",               "pm10"),
            "co_ppm":   _aq("carbon_monoxide",    "carbon_monoxide")  / 1145.0,
            "no2_ppb":  _aq("nitrogen_dioxide",   "nitrogen_dioxide") / 1.88,
            # Weather — instantaneous current reading
            "tmp":      _wx("temperature_2m"),
            "hmd":      _wx("relative_humidity_2m"),
            "hi":       _wx("apparent_temperature"),
            "wind_ms":  _wx("wind_speed_10m", 0.0) / 3.6,
            # Solar: today's mean (W/m² → kWh/m²/day)
            "solar_kwh_m2_day": solar_kwh,
            # Metadata
            "data_ts": data_ts,          # ISO timestamp of the current AQ reading
        }
        _LIVE_CACHE[key] = out if not math.isnan(out["tmp"]) else None
    except Exception as _e:
        # Silently cache None so callers fall back to sensor/climate-zone data
        _LIVE_CACHE[key] = None
    return _LIVE_CACHE[key]

def get_terrain(lat: float, lon: float):
    """Slope/relief from Open-Meteo elevation API (9-point ~1 km grid).
    Drives feasibility: no ground-mounted solar / heavy works on steep terrain."""
    key = (round(lat, 2), round(lon, 2))
    if key in _TERRAIN_CACHE: return _TERRAIN_CACHE[key]
    import requests as _r
    try:
        d = 0.009
        pts = [(lat,lon),(lat+d,lon),(lat-d,lon),(lat,lon+d),(lat,lon-d),
               (lat+d,lon+d),(lat-d,lon-d),(lat+d,lon-d),(lat-d,lon+d)]
        r = _r.get("https://api.open-meteo.com/v1/elevation",
                   params={"latitude": ",".join(str(p[0]) for p in pts),
                           "longitude": ",".join(str(p[1]) for p in pts)}, timeout=10).json()
        el = r["elevation"]; relief = max(el) - min(el)
        slope = ("flat" if relief < 15 else "rolling" if relief < 60
                 else "steep" if relief < 150 else "mountainous")
        _TERRAIN_CACHE[key] = {"elevation_m": round(el[0]), "relief_m_per_km": round(relief),
                               "slope_class": slope}
    except Exception:
        _TERRAIN_CACHE[key] = None
    return _TERRAIN_CACHE[key]

ZONE_SOLAR = {"Af":4.5,"Am":4.8,"Aw":5.2,"BWh":6.3,"BWk":5.5,"BSh":5.8,"BSk":5.0,
              "Cfa":4.3,"Cfb":2.9,"Dfa":3.6,"Dfb":3.2}

def get_population_density(lat: float, lon: float) -> float:
    """persons/km² — REAL data first: nearest OSM place node with a population
    tag (Overpass) → place-type typical density → gazetteer anchor → rural 300."""
    key = (round(lat, 2), round(lon, 2))
    if key in _POP_CACHE: return _POP_CACHE[key]
    dens = None
    try:
        q = (f'[out:json][timeout:15];node["place"~"city|town|suburb|village"]'
             f'(around:12000,{lat},{lon});out 30;')
        els = _rq_get_post("https://overpass-api.de/api/interpreter", q).json()["elements"]
        best = None
        for e in els:
            t = e.get("tags", {})
            score = (0 if t.get("population") else 1,
                     {"city":0,"suburb":1,"town":2,"village":3}.get(t.get("place"), 4))
            if best is None or score < best[0]: best = (score, t)
        if best:
            t = best[1]; place = t.get("place", "town")
            if t.get("population"):
                pop = float(re.sub(r"[^\d.]", "", t["population"]) or 0)
                typ_area = {"city":120,"suburb":10,"town":25,"village":6}.get(place, 25)
                dens = max(pop/typ_area, 50)
            else:
                dens = {"city":8000,"suburb":9000,"town":2500,"village":400}.get(place, 1000)
    except Exception:
        pass
    if dens is None:  # offline → gazetteer anchor with decay
        best, bd = 300.0, 1e9
        for name,(la,lo,co,kz,d_,rain) in GAZETTEER.items():
            dd_ = ((lat-la)**2+(lon-lo)**2)**0.5
            if dd_ < bd: best, bd = d_, dd_
        dens = best if bd < 0.15 else best*max(0.15, 1-bd/1.5) if bd < 1.5 else 300.0
    _POP_CACHE[key] = float(dens)
    return float(dens)

def get_annual_rainfall(lat: float, lon: float) -> float:
    """Annual rainfall (mm): REAL ERA5 12-month total for these exact coordinates
    → gazetteer city if within ~30 km → climate-zone typical value. Never the
    nearest-big-city guess that made Jaisalmer look like Delhi."""
    cn = fetch_climate_normals(lat, lon)
    if cn: return float(cn["annual_rain_mm"])
    best, bd = None, 1e9
    for name,(la,lo,co,kz,dens,rain) in GAZETTEER.items():
        d = ((lat-la)**2+(lon-lo)**2)**0.5
        if d < bd: best, bd = rain, d
    if bd < 0.3: return float(best)
    return float(ZONE_RAIN.get(get_climate_zone(lat, lon), 1000))

def classify_density(d: float) -> str:
    return ("Rural" if d < 150 else "Peri-urban" if d < 1000 else
            "Urban" if d < 5000 else "Dense Urban")

# ---------- Species database ----------
SPECIES_DB = [ # name, common, zone, role, co2 kg/yr, water, native_to, season, growth, layer
 ("Ficus benghalensis","Banyan","tropical","Keystone shade, carbon sink, bird habitat",28,"medium","India","Jun–Jul","slow-massive","canopy"),
 ("Azadirachta indica","Neem","tropical","Air purifier (NO2/SO2), medicinal, drought-hardy",22,"low","India","Jun–Aug","medium","canopy"),
 ("Terminalia arjuna","Arjun","tropical","Riparian stabiliser, carbon sink",25,"high","India","Jun–Jul","medium","canopy"),
 ("Neolamarckia cadamba","Kadamba","tropical","Fast carbon, pollinator support",24,"medium","India","Jun–Jul","fast","canopy"),
 ("Shorea robusta","Sal","tropical","Forest restoration dominant",26,"medium","India","Jun–Jul","slow","emergent"),
 ("Madhuca longifolia","Mahua","tropical","Livelihood, pollinator, dry-tolerant",21,"low","India","Jun–Jul","slow","canopy"),
 ("Butea monosperma","Palash","tropical","Pollinator (Feb–Mar bloom), N-fixing",17,"low","India","Jul–Aug","slow","sub-canopy"),
 ("Delonix regia","Krishnachura/Gulmohar","tropical","Avenue flowering, heat tolerant",18,"low","naturalized","Jun–Jul","fast","sub-canopy"),
 ("Bombax ceiba","Shimul/Silk cotton","tropical","Emergent layer, bird nesting",24,"medium","India","Jun–Jul","fast","emergent"),
 ("Artocarpus heterophyllus","Jackfruit","tropical","Food security + shade",21,"medium","India","Jun–Aug","medium","canopy"),
 ("Syzygium cumini","Jamun","tropical","Riparian, edible, dense shade",23,"medium","India","Jun–Jul","medium","canopy"),
 ("Bambusa balcooa","Bamboo (clumping)","tropical","Fastest carbon, soil binder, NOT for monoculture",35,"medium","India","Jun–Jul","very fast","sub-canopy"),
 ("Cassia fistula","Amaltas","tropical","Compact avenue tree for dense urban",15,"low","India","Jun–Jul","medium","sub-canopy"),
 ("Lagerstroemia speciosa","Pride of India/Jarul",ZONE_TO_SPECIES_KEY["Aw"],"Compact flowering, waterlogging-tolerant",14,"medium","India","Jun–Jul","medium","sub-canopy"),
 ("Pongamia pinnata","Karanja","tropical","N-fixing, biodiesel, coastal-tolerant",20,"low","India","Jun–Aug","medium","canopy"),
 ("Chrysopogon zizanioides","Vetiver grass","tropical","Bund/slope erosion control, phytoremediation",2,"low","India","Jun–Sep","fast","ground"),
 ("Nelumbo nucifera","Lotus","tropical","Aquatic bioremediation, biodiversity",1,"aquatic","India","Mar–Jun","fast","aquatic"),
 ("Typha angustifolia","Cattail/Hogla","tropical","Reed-bed water polishing",2,"aquatic","India","Jun–Sep","fast","aquatic"),
 ("Prosopis cineraria","Khejri","arid","Desert keystone, N-fixing fodder",15,"very low","India","Jul–Aug","slow","canopy"),
 ("Moringa oleifera","Drumstick","arid","Fast nutrition tree, drought-hardy",12,"very low","India","Jun–Aug","very fast","sub-canopy"),
 ("Ziziphus mauritiana","Ber/Indian jujube","arid","Fruit, hardy, pollinator",13,"very low","India","Jul–Aug","medium","sub-canopy"),
 ("Vachellia nilotica","Babul","arid","N-fixing, gum, fodder",16,"very low","India","Jul–Aug","medium","canopy"),
 ("Quercus robur","English oak","temperate","Keystone biodiversity (2300+ spp.), carbon",30,"medium","Europe","Nov–Mar","slow","canopy"),
 ("Tilia cordata","Small-leaved lime","temperate","Pollinator keystone, avenue",26,"medium","Europe","Nov–Mar","medium","canopy"),
 ("Platanus x acerifolia","London plane","temperate","Pollution-tolerant avenue standard",29,"medium","naturalized","Nov–Mar","fast","canopy"),
 ("Betula pendula","Silver birch","temperate","PM capture (leaf hairs), pioneer",18,"medium","Europe","Nov–Mar","fast","sub-canopy"),
 ("Sorbus aucuparia","Rowan","temperate","Bird forage, compact",12,"medium","Europe","Nov–Mar","medium","sub-canopy"),
 ("Croton megalocarpus","Croton","tropical","Highland shade + carbon (East Africa)",25,"medium","East Africa","Mar–May","fast","canopy"),
 ("Markhamia lutea","Nile tulip","tropical","Highland avenue, timber (East Africa)",18,"low","East Africa","Mar–May","fast","sub-canopy"),
 ("Chlorophytum comosum","Spider plant","global","Indoor/vertical-garden VOC removal",1,"low","Africa","any","fast","ground"),
 ("Spathiphyllum wallisii","Peace lily","global","Indoor VOC/formaldehyde removal",1,"medium","Americas","any","medium","ground"),
 ("Hedera helix","English ivy","temperate","NO2-absorbing green screens",2,"low","Europe","Sep–Nov","fast","ground"),
]
INVASIVE_BLACKLIST = ["Prosopis juliflora","Lantana camara","Eichhornia crassipes",
                      "Parthenium hysterophorus","Leucaena leucocephala","Acacia mearnsii"]

COUNTRY_TO_REGION = {"india":"India","bangladesh":"India","pakistan":"India","nepal":"India",
 "sri lanka":"India","kenya":"East Africa","ethiopia":"East Africa","tanzania":"East Africa",
 "united kingdom":"Europe","germany":"Europe","france":"Europe","russia":"Europe"}

def filter_species_by_climate(climate_zone: str, density_class: str, n: int = 12,
                              objective: str = "general", country: str = "") -> list:
    """Return suitable species records for zone + density, invasive-checked,
    preferring species native to the detected region."""
    key = ZONE_TO_SPECIES_KEY.get(climate_zone, "tropical")
    rows = [s for s in SPECIES_DB if s[2] in (key, "global")]
    region = COUNTRY_TO_REGION.get((country or "").strip().lower(), None)
    def native_penalty(s):
        if region is None or s[6] in ("naturalized", "global"): return 0
        return 0 if s[6] == region else 1
    if density_class == "Dense Urban":
        pref = {"sub-canopy","ground","aquatic"}
        rows.sort(key=lambda s: (native_penalty(s), s[9] not in pref, -s[4]))
    elif density_class == "Rural":
        rows.sort(key=lambda s: (native_penalty(s), s[9] not in {"emergent","canopy"}, -s[4]))
    else:
        rows.sort(key=lambda s: (native_penalty(s), -s[4]))
    if objective == "water":
        rows = [s for s in rows if s[9] in {"aquatic","ground"} or s[5] in {"high","aquatic"}] + rows
    rows = [s for s in rows if s[0] not in INVASIVE_BLACKLIST]
    seen, out = set(), []
    for s in rows:
        if s[0] in seen: continue
        seen.add(s[0]); out.append(s)
        if len(out) >= n: break
    return [dict(zip(["scientific","common","zone","role","co2_kg_yr","water",
                      "native_to","season","growth","layer"], s)) for s in out]

# ---------- Sensor loader ----------
def load_sensor_data(lat: float, lon: float, radius_deg: float = 0.05,
                     df: 'pd.DataFrame' = None) -> 'pd.DataFrame':
    df = sensor_df if df is None else df
    m = (df["LAT"].sub(lat).abs() <= radius_deg) & (df["LON"].sub(lon).abs() <= radius_deg)
    return df[m].copy()

# ---------- What-if simulation engine ----------
TOTAL_URBAN_AREA_HA = 20500  # KMC area ≈ 205 km²; overridden per location below

def simulate_intervention(baseline_aqi: float, trees_to_plant: int, area_hectares: float,
                          years: int, water_harvesting_m3: float = 0,
                          urban_area_ha: float = TOTAL_URBAN_AREA_HA) -> dict:
    """Project outcomes of a hypothetical intervention (medium confidence).
    AQI: ~0.3 points per 100 mature trees (urban canopy literature), capped at 30%.
    CO2: 22 kg/tree/yr default × growth ramp × 85% survival."""
    ramp = sum(min(0.1 + 0.1*y, 1.0) for y in range(years))   # sapling growth curve
    aqi_red = min((trees_to_plant/100)*0.3*min(years,10), baseline_aqi*0.30)
    co2_kg = trees_to_plant * 22 * ramp * 0.85
    return {"projected_aqi": round(max(0, baseline_aqi - aqi_red), 1),
            "aqi_reduction": round(aqi_red, 1),
            "co2_captured_tonnes": round(co2_kg/1000, 1),
            "cars_equivalent": int(co2_kg/1000/4.6/max(years,1)),
            "green_cover_increase_pct": round(area_hectares/urban_area_ha*100, 2),
            "water_saved_m3": int(water_harvesting_m3*years),
            "confidence": "medium"}

print("Tools ready ✓  (AQI, geocode, climate, population, species filter, loader, simulator)")
print("  e.g. compute_aqi(no2_ppm=0.04, co_ppm=8, dd=250) →", compute_aqi(no2_ppm=0.04, co_ppm=8, dd=250))


# ════════ 07_llm.py ════════
# ============================================================
# Cell 7: Pluggable LLM Backend
#
# Priority order (auto-detected at startup):
#   1. AMD vLLM   — OpenAI-compatible vLLM server on AMD Instinct GPU
#                   (set AMD_VLLM_URL + AMD_VLLM_MODEL env vars)
#   2. Ollama     — local model running via `ollama serve` (AMD ROCm or CPU)
#   3. HuggingFace Inference API — cloud fallback (requires HF_TOKEN)
#   4. None       — fully deterministic engine; all agents still work,
#                   narrative is un-polished but scientifically correct
#
# Every LLM call is timed and its token usage captured in LLM_LOG so the
# dashboard can display latency, token counts, and the active backend.
#
# Environment variables (set before running the notebook):
#   AMD_VLLM_URL    URL of the vLLM server  e.g. http://localhost:8000
#   AMD_VLLM_MODEL  Model name served by vLLM  e.g. meta-llama/Llama-3.1-8B-Instruct
#   AMD_VLLM_KEY    API key for the vLLM endpoint (empty string = no auth)
#   OLLAMA_URL      Ollama API base  (default: http://localhost:11434)
#   HF_TOKEN        HuggingFace user access token
#   HF_MODEL        HuggingFace model repo  (default: Mistral-7B-Instruct-v0.3)
# ============================================================
import time
import requests as _rq

# ── Backend configuration ────────────────────────────────────────────────────
#
# AMD vLLM — matches the AMD Instinct / MI300X workshop configuration.
#
# To start the vLLM server on an AMD GPU (run in a terminal):
#
#   VLLM_USE_TRITON_FLASH_ATTN=0 \
#   vllm serve Qwen/Qwen3-30B-A3B \
#       --served-model-name Qwen3-30B-A3B \
#       --api-key abc-123 \
#       --port 8000 \
#       --enable-auto-tool-choice \
#       --tool-call-parser hermes \
#       --trust-remote-code
#
# Monitor GPU utilisation while serving:
#   watch rocm-smi
#
# Defaults match the AMD workshop setup; override with env vars if needed:
AMD_VLLM_URL   = os.environ.get("AMD_VLLM_URL",   "http://localhost:8000")
AMD_VLLM_MODEL = os.environ.get("AMD_VLLM_MODEL", "Qwen3-30B-A3B")
AMD_VLLM_KEY   = os.environ.get("AMD_VLLM_KEY",   "abc-123")   # AMD workshop default key

# Ollama: local model server (works with AMD ROCm via ollama pull + ollama serve)
OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODELS = ["mistral:7b-instruct", "mistral", "llama3.1:8b", "llama3.1", "llama3"]

# HuggingFace Inference API: cloud fallback
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL  = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

# ── AMD vLLM setup helper ─────────────────────────────────────────────────────

def setup_amd_vllm(model="Qwen/Qwen3-30B-A3B", served_name="Qwen3-30B-A3B",
                   api_key="abc-123", port=8000):
    """Print the exact terminal command to launch vLLM on an AMD Instinct GPU,
    then probe the endpoint and reinitialise EcoGPT's LLM backend.

    Matches the AMD MI300X workshop configuration (Qwen3-30B-A3B, port 8000).

    Args:
        model       : HuggingFace model ID to download/serve
        served_name : Alias shown in /v1/models (used in API calls)
        api_key     : Bearer token for the local endpoint (any string works)
        port        : TCP port the vLLM server listens on

    Usage in notebook:
        setup_amd_vllm()          # defaults match AMD workshop
        setup_amd_vllm("Qwen/Qwen3-8B", served_name="Qwen3-8B")  # smaller model
    """
    cmd = (f"VLLM_USE_TRITON_FLASH_ATTN=0 \\\n"
           f"vllm serve {model} \\\n"
           f"    --served-model-name {served_name} \\\n"
           f"    --api-key {api_key} \\\n"
           f"    --port {port} \\\n"
           f"    --enable-auto-tool-choice \\\n"
           f"    --tool-call-parser hermes \\\n"
           f"    --trust-remote-code")
    print("── AMD vLLM launch command ──────────────────────────────")
    print(cmd)
    print("\nMonitor GPU (AMD ROCm):")
    print("  watch rocm-smi")
    print("\nOnce the server prints 'Application startup complete', run:")
    print("  global LLM; LLM = LLMBackend()   # to reinitialise EcoGPT's backend")
    print("─────────────────────────────────────────────────────────")

    # Also set env vars so LLMBackend() picks them up on re-init
    os.environ["AMD_VLLM_URL"]   = f"http://localhost:{port}"
    os.environ["AMD_VLLM_MODEL"] = served_name
    os.environ["AMD_VLLM_KEY"]   = api_key

    # Probe (non-fatal — the server may not be started yet)
    try:
        r = _rq.get(f"http://localhost:{port}/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"}, timeout=3)
        if r.status_code == 200:
            models = [m["id"] for m in r.json().get("data", [])]
            print(f"✅ vLLM server already running — models: {models}")
        else:
            print(f"⚠️  vLLM server responded with HTTP {r.status_code} (may still be loading)")
    except Exception:
        print("ℹ️  vLLM server not yet reachable — start it with the command above first")


# ── LLM call log ─────────────────────────────────────────────────────────────
# Each entry added by LLMBackend.generate():
#   {"ts": ISO-str, "backend": str, "model": str, "latency_ms": int,
#    "tokens_in": int, "tokens_out": int, "success": bool}
LLM_LOG = []


def llm_usage_summary():
    """Print a human-readable summary of all LLM calls made so far.

    Shows: backend used, total calls, total tokens (in/out), mean latency,
    and per-call detail.  Call this at the end of a notebook session to
    audit inference cost and performance."""
    if not LLM_LOG:
        print("No LLM calls logged yet.")
        return
    total_in  = sum(e["tokens_in"]  for e in LLM_LOG)
    total_out = sum(e["tokens_out"] for e in LLM_LOG)
    mean_lat  = sum(e["latency_ms"] for e in LLM_LOG) / len(LLM_LOG)
    ok        = sum(1 for e in LLM_LOG if e["success"])
    print(f"\n{'═'*62}")
    print(f"  EcoGPT LLM Usage Summary  ({len(LLM_LOG)} calls, {ok} succeeded)")
    print(f"{'═'*62}")
    print(f"  Backend    : {LLM_LOG[-1]['backend']}  ({LLM_LOG[-1]['model']})")
    print(f"  Tokens in  : {total_in:,}   Tokens out: {total_out:,}"
          f"   Total: {total_in+total_out:,}")
    print(f"  Avg latency: {mean_lat:.0f} ms   "
          f"Min: {min(e['latency_ms'] for e in LLM_LOG)} ms   "
          f"Max: {max(e['latency_ms'] for e in LLM_LOG)} ms")
    print(f"{'─'*62}")
    for i, e in enumerate(LLM_LOG, 1):
        status = "✅" if e["success"] else "❌"
        print(f"  [{i:2d}] {status} {e['ts'][:19]}  "
              f"{e['latency_ms']:5d} ms  "
              f"in={e['tokens_in']:4d}  out={e['tokens_out']:4d}  "
              f"{e['model']}")
    print(f"{'═'*62}\n")


class LLMBackend:
    """Auto-detects the best available LLM backend at startup.

    Call `backend.generate(system, user)` from any agent to get a polished
    text response.  Returns None (not an exception) when all backends fail —
    the deterministic scientific core always provides the fallback output.

    Attributes:
        kind  (str): Active backend — "amd_vllm" | "ollama" | "hf" | "none"
        model (str): Model name/path currently in use
    """

    def __init__(self):
        self.kind  = "none"
        self.model = None

        # ── 1. AMD vLLM (highest priority when configured) ────────────────
        # AMD provides a vLLM server with OpenAI-compatible /v1/chat/completions.
        # Set AMD_VLLM_URL + AMD_VLLM_MODEL to activate.
        if AMD_VLLM_URL and AMD_VLLM_MODEL:
            try:
                headers = {}
                if AMD_VLLM_KEY:
                    headers["Authorization"] = f"Bearer {AMD_VLLM_KEY}"
                # Probe the /v1/models endpoint to confirm the server is alive
                probe = _rq.get(f"{AMD_VLLM_URL.rstrip('/')}/v1/models",
                                headers=headers, timeout=4)
                if probe.status_code == 200:
                    self.kind  = "amd_vllm"
                    self.model = AMD_VLLM_MODEL
            except Exception:
                pass  # vLLM server not reachable; try next backend

        # ── 2. Ollama (local, AMD ROCm-accelerated if available) ──────────
        if self.kind == "none":
            try:
                tags      = _rq.get(f"{OLLAMA_URL}/api/tags", timeout=2).json()
                available = [m["name"] for m in tags.get("models", [])]
                for want in OLLAMA_MODELS:
                    hit = next((a for a in available if a.startswith(want)), None)
                    if hit:
                        self.kind, self.model = "ollama", hit
                        break
                # Fallback: take whatever model is loaded
                if self.kind == "none" and available:
                    self.kind, self.model = "ollama", available[0]
            except Exception:
                pass

        # ── 3. HuggingFace Inference API (cloud, no local GPU needed) ─────
        if self.kind == "none" and HF_TOKEN:
            self.kind, self.model = "hf", HF_MODEL

        # ── Report active backend ──────────────────────────────────────────
        _backend_label = {
            "amd_vllm": "AMD vLLM",
            "ollama":   "Ollama (local)",
            "hf":       "HuggingFace Inference API",
            "none":     "None",
        }[self.kind]
        _model_info = f" ({self.model})" if self.model else ""
        print(f"LLM backend: {_backend_label}{_model_info}"
              + ("" if self.kind != "none" else
                 "  → deterministic engine only (fully functional, narrative un-polished)"))

    # ─────────────────────────────────────────────────────────────────────────
    def generate(self, system: str, user: str,
                 max_tokens: int = 900, temperature: float = 0.4) -> "str | None":
        """Send a chat-completion request to the active backend.

        Logs every call to the module-level LLM_LOG list with:
          - backend / model name
          - latency in milliseconds
          - prompt tokens (tokens_in) and completion tokens (tokens_out)
          - success flag

        Args:
            system      : System prompt (agent persona + instructions)
            user        : User turn (context + question)
            max_tokens  : Maximum tokens to generate (default 900)
            temperature : Sampling temperature 0–1 (default 0.4)

        Returns:
            Stripped response string, or None on failure (callers use
            deterministic output as fallback).
        """
        import datetime
        _log = {
            "ts":          datetime.datetime.now().isoformat(),
            "backend":     self.kind,
            "model":       self.model or "none",
            "latency_ms":  0,
            "tokens_in":   0,
            "tokens_out":  0,
            "success":     False,
        }
        t0 = time.time()
        result = None
        try:
            # ── AMD vLLM: OpenAI-compatible /v1/chat/completions ──────────
            if self.kind == "amd_vllm":
                headers = {"Content-Type": "application/json"}
                if AMD_VLLM_KEY:
                    headers["Authorization"] = f"Bearer {AMD_VLLM_KEY}"
                resp = _rq.post(
                    f"{AMD_VLLM_URL.rstrip('/')}/v1/chat/completions",
                    headers=headers,
                    json={
                        "model":       self.model,
                        "max_tokens":  max_tokens,
                        "temperature": temperature,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                    },
                    timeout=180,
                ).json()
                result = resp["choices"][0]["message"]["content"].strip()
                # vLLM returns OpenAI-style usage object
                usage = resp.get("usage", {})
                _log["tokens_in"]  = usage.get("prompt_tokens",     0)
                _log["tokens_out"] = usage.get("completion_tokens", 0)

            # ── Ollama: native /api/chat endpoint ─────────────────────────
            elif self.kind == "ollama":
                resp = _rq.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model":   self.model,
                        "stream":  False,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                    timeout=180,
                ).json()
                result = resp["message"]["content"].strip()
                # Ollama reports eval_count (output) and prompt_eval_count (input)
                _log["tokens_in"]  = resp.get("prompt_eval_count", 0)
                _log["tokens_out"] = resp.get("eval_count",        0)

            # ── HuggingFace Inference API: OpenAI-compat /v1/chat ─────────
            elif self.kind == "hf":
                resp = _rq.post(
                    f"https://api-inference.huggingface.co/models/{self.model}"
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    json={
                        "model":       self.model,
                        "max_tokens":  max_tokens,
                        "temperature": temperature,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                    },
                    timeout=120,
                ).json()
                result = resp["choices"][0]["message"]["content"].strip()
                usage = resp.get("usage", {})
                # HF free-tier Inference API often omits the usage object entirely.
                # Fall back to a character-based estimate (÷4 ≈ GPT-style token ratio).
                _log["tokens_in"]  = (usage.get("prompt_tokens")     or
                                      len(system + user) // 4)
                _log["tokens_out"] = (usage.get("completion_tokens") or
                                      len(result) // 4)

            _log["success"] = result is not None

        except Exception as _e:
            print(f"  LLM call failed ({self.kind}): {_e} — deterministic output used")

        _log["latency_ms"] = int((time.time() - t0) * 1000)
        LLM_LOG.append(_log)

        # Print a one-line telemetry receipt after each call
        status = "✅" if _log["success"] else "❌"
        print(f"  {status} LLM [{self.kind}] {_log['model']}  "
              f"latency={_log['latency_ms']} ms  "
              f"tokens in={_log['tokens_in']} out={_log['tokens_out']}")
        return result


# Instantiate the singleton backend (probes all endpoints at import time)
LLM = LLMBackend()


# ════════ 08_agents.py ════════
# ============================================================
# Cell 8: The 8 EcoGPT Agents
#   Each agent = system prompt (for LLM/ADK use) + deterministic
#   scientific core (always runs) + RAG grounding.
# ============================================================

SYSTEM_PROMPTS = {
"data_ingestion": """You are the Data Ingestion Agent for EcoGPT. Accept a latitude/longitude or city name.
Load and filter sensor CSV data within a 0.05° radius. Compute mean/max/min/std per sensor channel.
Derive composite AQI from NO2, CO, PM (RAWPM or dust-density proxy) and MQ135 using EPA breakpoints.
Classify AQI: Good 0-50, Moderate 51-100, USG 101-150, Unhealthy 151-200, Very Unhealthy 201-300, Hazardous >300.
Flag anomalies >2σ above mean. Return structured JSON. State assumptions clearly when data is sparse.""",
"rag_retrieval": """You are the RAG Retrieval Agent for EcoGPT. Given environmental context, retrieve top-K relevant
chunks covering species, pollution mitigation, water management, soil, urban greening, carbon data, and regulation.
Always return source, relevance score, and the retrieving query. Never hallucinate; say explicitly if nothing is found.""",
"pollution": """You are the Air Quality and Pollution Specialist for EcoGPT. Assess severity from AQI, CO, NO2, VOC,
MQ2, MQ7, MQ135, PM. Identify sources: traffic (high NO2+CO), industrial (high VOC+MQ135), biomass burning (high MQ2+CO), or mixed.
Recommend short-term (24-72h), medium-term (1-6mo), long-term (1-5yr) actions. Specify bio-filter species per pollutant
(NO2: Ficus benjamina, Hedera helix; PM2.5: Tillandsia usneoides, Ficus elastica; VOC: Chlorophytum comosum, Spathiphyllum wallisii).
Quantify expected AQI improvement from green interventions. Flag heat-island risk if HI exceeds TMP by >3°C.""",
"plantation": """You are the Plantation and Biodiversity Specialist for EcoGPT. Recommend 8-12 named species with scientific
name, common name, suitability, ecological role, season, growth, CO2 rate, water need, native status. Design a plantation plan:
trees/ha scaled to population density, spatial arrangement, priority zones, 5-layer canopy strategy. Never recommend invasive
species (Prosopis juliflora, Lantana camara, Eichhornia crassipes). Always polyculture. Scale by density:
<500/km² large canopy restoration; 500-5000 urban park + avenue; >5000 vertical gardens, rooftop greening, compact flowering trees.""",
"water": """You are the Water Conservation Specialist for EcoGPT. Assess water stress from humidity, temperature, climate-zone
rainfall. Recommend water-body restoration (desilting, bunding, riparian buffers with Vetiver/Bamboo/Typha, lotus/lily bioremediation,
bioswales, fish stocking), rainwater harvesting with area calculations, check dams, retention ponds, irrigation efficiency.
Quantify litres saved per year and groundwater improvement estimates.""",
"urban": """You are the Urban Planning and Population Specialist for EcoGPT. Classify density (Rural <150, Peri-urban 150-1000,
Urban 1000-5000, Dense Urban >5000 /km²). Compute green-space deficit vs WHO 9 m²/capita. Recommend land-use optimization that
displaces no residents: institutional land, medians, buffers, riverbanks, rooftops; Miyawaki micro-forests for dense areas.
Recommend green corridors, permeable pavement, cool roofs, bio-retention cells. Estimate load reduction per intervention.""",
"carbon": """You are the Carbon Sequestration Estimation Specialist for EcoGPT. Use IPCC Tier-1 style factors and i-Tree
species data. Apply growth ramp (saplings ~10% of mature rate, full by year ~10) and 15% first-3-year mortality.
Project years 1/5/10/25. Compare against local emission load. Report tonnes CO2e/yr, cars-equivalent (4.6 t/car/yr),
households-equivalent. State confidence: High species-level / Medium genus / Low zone-average.""",
"synthesis": """You are the Synthesis Agent for EcoGPT. Merge all specialist JSON outputs, resolving contradictions by
deferring to the most data-grounded value. Output exactly: Environmental Assessment; Key Risks; Recommended Plant Species table;
Plantation Plan; Pollution Reduction Actions (short/medium/long); Soil Improvement; Water Conservation; Biodiversity Enhancement;
Carbon Sequestration table (yr 1/5/10/25); Predicted Environmental Impact; Priority Action Items ranked 1-10; What-If Simulation
with 2-3 scenarios. Calibrate tone to user type. Never fabricate species or figures; mark estimates "(estimated)" and assumptions "(assumed)".""",
}

# ---------------- Agent 1: Data Ingestion ----------------
def run_data_ingestion(lat, lon, radius=0.05, label=None):
    geo = reverse_geocode(lat, lon)
    # Prefer the user's own place name over reverse-geocoded suburb names
    # (e.g. 22.557,88.494 reverse-geocodes to "Newtown" — keep "Kolkata" if given)
    if label and not re.match(r"^\(", label):
        geo["city"] = label
    dens = get_population_density(lat, lon)
    local = load_sensor_data(lat, lon, radius)
    assumptions = []
    if len(local) < 10:  # widen search before giving up — nearby sensors still representative
        for r2 in (radius*3, radius*8):
            wider = load_sensor_data(lat, lon, r2)
            if len(wider) >= 10:
                local = wider
                assumptions.append(f"No readings within {radius}°; using {len(wider)} readings "
                                   f"from sensors within ~{r2*111:.0f} km (assumed representative)")
                break
    # Data priority: REAL uploaded IoT readings > LIVE per-location feeds >
    # synthetic demo rows > climate-zone model. Synthetic data never masks reality.
    if len(local) and "source" in local.columns:
        _real = local[~local["source"].astype(str).str.startswith("synthetic_")]
    else:
        _real = local
    use_real = len(_real) >= 10
    if use_real:
        local = _real
    live = fetch_live_environment(lat, lon)
    kz = geo["climate_zone"]
    use_sensors = use_real or (live is None and len(local) >= 10)
    live_overlay = False
    if (not use_sensors) and live is not None:
        # REAL location-specific data — different for every place on Earth
        data_source = ("LIVE per-location feeds: Open-Meteo CAMS air quality + ERA5 weather "
                       "(real data for these exact coordinates)")
        assumptions.append("No IoT sensors here — live satellite/model data fetched for this exact location")
        assumptions.append("VOC/MQ channels not available from satellite feeds — "
                           "source apportionment limited to NO2/CO/PM (assumed)")
        stats = {"TMP":{"mean":live["tmp"]},"HMD":{"mean":live["hmd"]},"HI":{"mean":live["hi"]},
                 "CO":{"mean":live["co_ppm"]},"NO2":{"mean":live["no2_ppb"]/1000.0},
                 "VOC":{"mean":0.0},"RAWPM":{"mean":live["pm25"]},"DD":{"mean":live["pm10"]},
                 "MQ2":{"mean":0.0},"MQ7":{"mean":0.0},"MQ135":{"mean":0.0},"C2H5OH":{"mean":0.0}}
        completeness, anomalies = 0.85, []
    elif not use_sensors:
        data_source = "climate-zone model (offline fallback — all values estimated)"
        assumptions.append(f"No sensor coverage at ({lat:.3f},{lon:.3f}) and offline — "
                           "climate-zone modelled values used (estimated)")
        base = {"Aw":(31,65,8,0.005,40,120,200),"Af":(28,80,5,0.004,35,90,150),
                "BWh":(36,30,9,0.006,50,160,300),"BSh":(33,40,9,0.006,45,150,280),
                "Cfa":(22,65,4,0.004,30,60,90),"Cfb":(14,75,3,0.003,25,40,60),
                "Dfb":(10,70,3,0.003,25,45,70)}.get(kz,(25,60,5,0.004,35,80,120))
        t,h,co,no2,voc,pm,dd = base
        stats = {"TMP":{"mean":t},"HMD":{"mean":h},"CO":{"mean":co},"NO2":{"mean":no2},
                 "VOC":{"mean":voc},"RAWPM":{"mean":pm},"DD":{"mean":dd},
                 "HI":{"mean":heat_index_c(t,h)},"MQ2":{"mean":3},"MQ7":{"mean":3},
                 "MQ135":{"mean":5},"C2H5OH":{"mean":20}}
        completeness, anomalies = 0.0, []
    else:
        data_source = (f"local IoT sensor network ({len(local):,} real readings)" if use_real else
                       f"synthetic demo sensor data ({len(local):,} rows, calibrated to reference CSV) "
                       "— connect internet for live feeds or add real sensors")
        stats = {c: {"mean": float(local[c].mean()), "max": float(local[c].max()),
                     "min": float(local[c].min()), "std": float(local[c].std() or 0)}
                 for c in SENSOR_COLS if c in local}
        # STALENESS GUARD: old sensor archives must not masquerade as current conditions.
        last_t = pd.to_datetime(local["time"]).max()
        age_days = int((pd.Timestamp.now() - last_t).days)
        live_overlay = False
        if age_days > 30 and live is not None:
            live_overlay = True
            for c, v in [("TMP",live["tmp"]),("HMD",live["hmd"]),("HI",live["hi"]),
                         ("CO",live["co_ppm"]),("NO2",live["no2_ppb"]/1000.0),
                         ("RAWPM",live["pm25"]),("DD",live["pm10"])]:
                stats[c] = {**stats.get(c, {}), "mean": float(v)}
            data_source += (f" + LIVE overlay (sensor archive ends {last_t.date()}, "
                            f"{age_days} days old — current weather/air from Open-Meteo)")
            assumptions.append(f"IoT readings are {age_days} days old — current conditions "
                               "taken from live feeds; sensor MQ/VOC chemistry retained for "
                               "pollution-source fingerprinting (assumed still representative)")
        elif age_days > 30:
            assumptions.append(f"Sensor archive ends {last_t.date()} ({age_days} days old) and "
                               "offline — treat absolute values as historical (estimated)")
        completeness = float(local[SENSOR_COLS].notna().mean().mean())
        anomalies = []
        for c in ["CO","NO2","VOC","RAWPM","DD","MQ2","MQ135"]:
            if c in stats and stats[c].get("std", 0) > 0:
                thresh = stats[c]["mean"] + 2*stats[c]["std"]
                n_anom = int((local[c] > thresh).sum())
                if n_anom: anomalies.append(f"{c}: {n_anom} readings >2σ (max {stats[c]['max']:.1f})")
        if local["RAWPM"].notna().mean() < 0.2:
            assumptions.append("RAWPM mostly missing — PM2.5 proxied from dust density ×0.4 (assumed)")
    pm_val = stats.get("RAWPM",{}).get("mean", float("nan"))
    if (not live_overlay) and len(local) >= 10 and local.get("RAWPM") is not None \
            and local["RAWPM"].notna().mean() < 0.2:
        pm_val = float("nan")
    aqi = compute_aqi(no2_ppm=stats.get("NO2",{}).get("mean"),
                      co_ppm=stats.get("CO",{}).get("mean"),
                      pm25=None if math.isnan(pm_val) else pm_val,
                      mq135=stats.get("MQ135",{}).get("mean"),
                      dd=stats.get("DD",{}).get("mean"))
    # optional Kaggle city-level baseline (Cell 2b): override when nothing better,
    # cross-check otherwise
    if "kaggle_city_baseline" in globals():
        kb_ = kaggle_city_baseline(geo["city"])
        if kb_ and not math.isnan(kb_.get("aqi", float("nan"))):
            if len(local) < 10 and live is None:
                aqi = {"score": round(kb_["aqi"],1), "category": aqi_category(kb_["aqi"]),
                       "dominant": "PM2.5", "sub_indices": {},
                       "notes": ["city-level Kaggle baseline used (static dataset)"]}
                data_source = "Kaggle global air-pollution city baseline (static)"
            else:
                aqi["notes"].append(f"Kaggle city baseline cross-check: AQI {kb_['aqi']:.0f}")
    # Solar/wind: prefer ANNUAL ERA5 means (seasonal-bias-free) → 7-day live → zone typical
    _cn = fetch_climate_normals(lat, lon)
    solar = ((_cn or {}).get("solar_kwh_m2_day") or
             (live["solar_kwh_m2_day"] if live else None) or ZONE_SOLAR.get(kz, 4.5))
    wind = ((_cn or {}).get("wind_ms") or (live["wind_ms"] if live else None) or 3.5)
    return {"location": {"city": geo["city"], "lat": lat, "lon": lon,
                         "country": geo["country"], "climate_zone": geo["climate_zone"],
                         "climate_name": CLIMATE_ZONE_NAMES.get(geo["climate_zone"], geo["climate_zone"])},
            "population_density": round(dens), "density_class": classify_density(dens),
            "aqi": aqi,
            "temperature_avg": round(stats["TMP"]["mean"],1), "humidity_avg": round(stats["HMD"]["mean"],1),
            "heat_index_avg": round(stats["HI"]["mean"],1),
            "co_avg": round(stats["CO"]["mean"],1), "no2_avg": round(stats["NO2"]["mean"],3),
            "voc_avg": round(stats["VOC"]["mean"],1),
            "pm_avg": round(stats.get("DD",{}).get("mean",0)*0.4,1),
            "data_source": data_source,
            "solar_kwh_m2_day": round(solar, 2), "wind_ms": round(wind, 1),
            "readings_used": len(local), "anomalies": anomalies,
            "data_completeness": round(completeness,2), "assumptions": assumptions,
            "stats": {k:{kk:round(vv,2) for kk,vv in v.items()} for k,v in stats.items()},
            # Store the raw live-environment dict so downstream code can access
            # the data timestamp and other metadata (e.g. for the dashboard footer)
            "live_env": live}

# ---------------- Agent 3: Pollution ----------------
def run_pollution_agent(ctx):
    s = ctx["stats"]
    def g(c):
        v = s.get(c, {}).get("mean", 0) or 0
        return 0.0 if (isinstance(v, float) and math.isnan(v)) else v
    sigs = {"traffic": (g("NO2")*120 + g("CO")*0.6),
            "industrial": (g("VOC")*0.9 + g("MQ135")*4),
            "biomass": (g("MQ2")*8 + g("CO")*0.4)}
    top = max(sigs, key=sigs.get)
    spread = sorted(sigs.values(), reverse=True)
    source = top if spread[0] > spread[1]*1.25 else "mixed"
    heat_island = ctx["heat_index_avg"] - ctx["temperature_avg"] > 3
    chunks = retrieve(build_retrieval_query("pollution", ctx), 3)
    cat = ctx["aqi"]["category"]
    score = ctx["aqi"]["score"]
    # ---- actions assembled from severity + detected source mix (not static lists) ----
    hedge_sp = filter_species_by_climate(ctx["location"]["climate_zone"], ctx["density_class"],
                                         3, country=ctx["location"].get("country",""))
    hedge = ", ".join(s["common"] for s in hedge_sp)
    short, medium, longt = [], [], []
    if score > 150:
        short += ["Public health advisory: N95 masks, limit outdoor exertion at peak hours",
                  "Halt construction & demolition 72h; water-sprinkle arterial roads twice daily"]
    elif score > 100:
        short += ["Sensitive-group advisory (children, elderly, cardio-respiratory patients)",
                  "Mechanised sweeping + water sprinkling on dust hotspots"]
    else:
        short += ["Maintain monitoring; publish daily AQI bulletins to sustain good air"]
    if source in ("traffic", "mixed"):
        short += ["Traffic demand management: odd-even / heavy-vehicle daytime restrictions, "
                  "public-transport fare incentives"]
        medium += [f"3-row green barrier hedges ({hedge}) along the worst arterials",
                   "Junction redesign + idling-engine enforcement at choke points"]
        longt += ["EV transition zone: e-bus depots, charging mandates in new buildings"]
    if source in ("industrial", "mixed"):
        short += ["Spot-check stack emissions at red-category units; suspend violators"]
        medium += ["Continuous emission monitoring on major stacks; VOC capture at solvent users"]
        longt += ["500 m industrial buffer greenbelts; relocate non-compliant units"]
    if source in ("biomass", "mixed"):
        short += ["Anti-open-burning patrols; fine refuse/leaf burning"]
        medium += ["LPG/electric conversion for street-food and small eateries; "
                   "decentralised composting so green waste is never burnt"]
    medium += ["Filtered ventilation retrofits in schools and clinics in hotspot zones"]
    longt += ["Urban forestry corridors connecting parks at ≤500 m spacing (3-30-300 rule)",
              "Green building code: cool roofs (albedo >0.65) + rooftop gardens"]
    if heat_island:
        longt += ["Heat-island programme: 30% canopy target + cool-roof retrofits on public buildings"]
    biofilters = {"NO2": ["Ficus benjamina","Hedera helix (green screens)","Azadirachta indica"],
                  "PM2.5/dust": ["Ficus elastica","Tillandsia usneoides","Neem hedgerows","Betula pendula"],
                  "VOC": ["Chlorophytum comosum","Spathiphyllum wallisii","Epipremnum aureum"]}
    return {"severity": cat, "aqi_score": ctx["aqi"]["score"], "dominant_pollutant": ctx["aqi"]["dominant"],
            "source_apportionment": {"classification": source,
                "signals": {k: round(v,1) for k,v in sigs.items()}},
            "heat_island_flag": heat_island,
            "heat_island_delta": round(ctx["heat_index_avg"]-ctx["temperature_avg"],1),
            "actions": {"short_term_24_72h": short, "medium_term_1_6mo": medium,
                        "long_term_1_5yr": longt},
            "biofilter_species": biofilters,
            "expected_improvement": "Street-canyon green walls: up to 40% NO2 / 60% PM locally; "
                "citywide canopy at 30%: 2-10% PM2.5 reduction + 1-3°C cooling (Pugh et al., estimated)",
            "grounding": [{"id":c["id"],"source":c["source"],"relevance":c["relevance"]} for c in chunks]}

# ---------------- Agent 4: Plantation & Biodiversity ----------------
def run_plantation_agent(ctx):
    dens_class = ctx["density_class"]
    kz_ = ctx["location"]["climate_zone"]
    arid = kz_.startswith("B")
    species = filter_species_by_climate(kz_, dens_class, 12,
                                        country=ctx["location"].get("country", ""))
    trees_ha = {"Rural": 400, "Peri-urban": 250, "Urban": 150, "Dense Urban": 100}[dens_class]
    arrangement = {"Rural": "Block restoration + farm bunds (agroforestry rows at 10 m)",
        "Peri-urban": "Cluster planting in commons + avenue rows on connecting roads",
        "Urban": "Avenue planting (8-10 m spacing) + pocket parks + biodiversity corridors",
        "Dense Urban": "Miyawaki micro-forests (3 saplings/m² on ≥30 m² plots), vertical gardens, "
                       "rooftop greening, compact flowering avenues"}[dens_class]
    if arid:  # DESERT MODE — dense planting is irresponsible in <500 mm zones
        trees_ha = {"Rural": 100, "Peri-urban": 80, "Urban": 60, "Dense Urban": 50}[dens_class]
        arrangement = ("Desert-appropriate: shelterbelt/windbreak rows perpendicular to prevailing "
                       "wind (3-row: tall Khejri/Babul centre, shrubs outside), micro-catchment "
                       "basins (half-moon bunds) per tree, drip irrigation in establishment years, "
                       "NO dense Miyawaki blocks — water budget cannot support them")
    chunks = retrieve(build_retrieval_query("plantation", ctx), 4)
    # ---- canopy layers derived from the ACTUAL species selected for this location ----
    layers = {}
    for s in species:
        layers.setdefault(s["layer"], []).append(s["scientific"])
    canopy_layers = {lyr: ", ".join(names[:3]) for lyr, names in layers.items()}
    for lyr in ("emergent","canopy","sub-canopy","shrub","ground"):
        canopy_layers.setdefault(lyr, "— (extend species DB for this layer in this climate zone)")
    if arid:
        priority_zones = ["Settlement windward edges (shelterbelts vs dust)",
                          "Around water points & khadin/check-dam catchments",
                          "Homestead & institutional compounds (greywater-fed)",
                          "Roadside single rows with micro-catchments"]
    else:
        priority_zones = {
        "Rural": ["Degraded commons & village forest blocks","Farm bunds (agroforestry rows)",
                  "Riverbanks & pond margins","School and panchayat/communal grounds"],
        "Peri-urban": ["Connecting-road avenues","Commons & grazing-land edges",
                       "Water-body bunds","Institutional campuses","Peri-urban wetland buffers"],
        "Urban": ["Roadsides & medians","Water-body bunds & riparian belts",
                  "Institutional campuses (schools/hospitals/depots)","Pocket parks on vacant lots"],
        "Dense Urban": ["Vacant municipal lots → Miyawaki plots","Rooftops & flyover pillars",
                        "Roadside verges & medians","Institutional campuses","Canal banks"],
    }[dens_class]
    return {"species": species, "trees_per_hectare": trees_ha,
            "spatial_arrangement": arrangement,
            "priority_zones": priority_zones,
            "canopy_layers": canopy_layers,
            "avoid": INVASIVE_BLACKLIST,
            "polyculture_rule": "No species >15% of total; ≥10 species/ha",
            "grounding": [{"id":c["id"],"source":c["source"],"relevance":c["relevance"]} for c in chunks]}

# ---------------- Agent 5: Water ----------------
REGION_FISH = {"India": "Rohu, Catla, Mrigal (Indian major carps)",
               "East Africa": "native Oreochromis/Labeo species (consult fisheries dept)",
               "Europe": "native cyprinids (roach, rudd, tench) — agency approval required"}

def run_water_agent(ctx):
    h, t = ctx["humidity_avg"], ctx["temperature_avg"]
    L = ctx["location"]
    rain = get_annual_rainfall(L["lat"], L["lon"])
    _cn = fetch_climate_normals(L["lat"], L["lon"])
    rain_source = ("measured: ERA5 last-12-month total for these coordinates" if _cn
                   else "climate-zone estimate (offline)")
    # rainfall-aware stress: arid rainfall dominates the humidity signal
    stress = ("High (dry + hot: evaporation stress)" if h < 40 and t > 35 else
              "High (arid climate)" if h < 40 else
              "Moderate" if h <= 70 else "Low (humid — drainage & waterlogging priority)")
    if rain < 500 and not stress.startswith("High"):
        stress = f"High (only {rain:.0f} mm rain/yr — scarcity overrides humidity signal)"
    region = COUNTRY_TO_REGION.get((L.get("country") or "").strip().lower())
    fish = REGION_FISH.get(region, "native fish per local fisheries authority (never exotics)")
    # riparian palette from the species DB for THIS climate zone
    rip = [s["scientific"] for s in filter_species_by_climate(
        L["climate_zone"], "Rural", 12, country=L.get("country","")) if s["water"] in ("high","aquatic")]
    rip = ", ".join(rip[:3]) if rip else "deep-rooted native riparian species"
    roof, persons = 100, 5
    harvest_l = roof * rain * 0.8
    chunks = retrieve(build_retrieval_query("water", ctx), 3)
    plan = ["Intercept/divert sewage inflows BEFORE physical works",
            "Dry-season desilting to original bed; reuse tested silt on bunds",
            f"Vetiver hedgerows on bunds; 10-30 m riparian buffer ({rip})",
            "Reed beds (Typha/Phragmites) at inlets; floating-leaf natives ≤30% surface",
            "Bioswales + constructed wetland for stormwater first-flush",
            f"Restock {fish} once dissolved oxygen >4 mg/L"]
    if L["climate_zone"].startswith("B"):
        # DESERT MODE — wetland-restoration playbook does not apply at this rainfall
        plan = [f"Water budget is the binding constraint ({rain:.0f} mm/yr) — every action below "
                "is harvest-and-conserve, not wetland restoration",
                "Khadins/check dams on ephemeral drainage lines to capture flash-flow",
                "Tanka/kund covered cisterns at homesteads (evaporation-proof storage)",
                "Recharge existing wells/beris from rooftop + courtyard catchments",
                "Greywater reuse mandatory for all tree irrigation",
                "Drip irrigation only; mulch every planting basin (50%+ evaporation cut)",
                "Protect any existing oasis/wetland absolutely — no extraction expansion",
                "Dune/dust control: shelterbelts + grass checkerboards on mobile sand"]
    elif L["climate_zone"].startswith("A") or region == "India":
        plan.insert(4, "Remove Eichhornia (water hyacinth) fully — weevil biocontrol + composting")
    if stress.startswith("High") and not L["climate_zone"].startswith("B"):
        plan = ["PRIORITY: check dams + percolation ponds for groundwater recharge",
                "Mulch + drip for all new plantations (no flood irrigation)"] + plan
    elif stress.startswith("Low"):
        plan = ["PRIORITY: drainage management — permeable paving, retention ponds, "
                "wetland buffers against waterlogging/flood"] + plan
    return {"stress_level": stress, "annual_rainfall_mm": int(rain),
        "rainfall_source": rain_source,
        "restoration_plan": plan,
        "rainwater_harvesting": {
            "per_100m2_roof_litres_yr": int(harvest_l),
            "household_demand_coverage_pct": round(100*harvest_l/(persons*135*365),1),
            "recharge_structures": "1-2 m³ percolation pit per 100 m² roof; boundary trenches; defunct-borewell shafts",
            "groundwater_response": "0.3-1.0 m table rise in 3-5 monsoons at ward scale (CGWB pilots, estimated)"},
        "irrigation": ["Flood→drip conversion saves ~3-4 ML/ha/yr (90% vs 40% efficiency)",
                       "Soil-moisture-sensor scheduling: further 15-25% saving"],
        "quantified_savings_l_yr": {"rwh_per_1000_households": int(harvest_l*1000),
                                    "drip_per_ha": 3_500_000},
        "grounding": [{"id":c["id"],"source":c["source"],"relevance":c["relevance"]} for c in chunks]}

# ---------------- Agent 6: Urban Planning ----------------
def run_urban_agent(ctx):
    d = ctx["population_density"]; dc = ctx["density_class"]
    deficit_m2_km2 = d * 9                      # WHO 9 m²/capita per km² of city
    green_needed_ha_km2 = deficit_m2_km2 / 10_000
    chunks = retrieve(build_retrieval_query("urban", ctx), 3)
    city = ctx["location"]["city"].lower()
    note = ("Current green cover should be measured via Sentinel-2/MODIS NDVI "
            "(see satellite layer in the dashboard, Cell 11)")
    if "kolkata" in city:
        note += "; Kolkata reference ≈2 m²/capita vs WHO 9 → ~7 m²/person deficit (estimated)"
    co2_pc = {"india":2.0,"bangladesh":0.7,"kenya":0.4,"nigeria":0.6,"united kingdom":4.7,
              "germany":7.9,"united states":14.7,"uae":21.8,"japan":8.5,"brazil":2.2,
              "singapore":8.9,"egypt":2.5,"russia":11.4,"australia":15.0
              }.get((ctx["location"].get("country") or "").lower(), 4.7)
    trees_per_1000 = int(1000 * co2_pc * 1000 / 22 / 1000)  # trees to offset 1% would be /100
    return {"density_per_km2": d, "classification": dc,
        "who_green_norm_m2pc": 9,
        "required_green_ha_per_km2": round(green_needed_ha_km2, 1),
        "per_capita_co2_t": co2_pc,
        "trees_per_1000_residents_full_offset": trees_per_1000,
        "note": note,
        "greening_without_displacement": ["Institutional campuses (15-30% of urban land)",
            "Road medians & verges (avenue planting)","Canal/river banks","Industrial buffers",
            "Rooftops (10-20% of plan area feasible)","Parking lots (40% shade-tree mandate)"],
        "dense_urban_toolkit": ["Miyawaki micro-forests: ≥90 trees per 30 m² plot",
            "Vertical gardens on flyover pillars (1 m² ≈ 2.3 kg CO2/yr + PM capture)",
            "Cool roofs (albedo >0.65) + 30% canopy target → 1-3°C ambient cooling",
            "Permeable pavement in markets (70-90% runoff cut)",
            "Bio-retention cells every 300 m of storm drain"],
        "co2_honesty_note": f"At ~{co2_pc} t CO2/person/yr here, ~{max(1,int(co2_pc*1000/22/100))*10} "
            "mature trees per 1000 residents offset only ~1% of their emissions — urban forests are "
            "for air, heat & habitat; pair with energy transition (estimated)",
        "grounding": [{"id":c["id"],"source":c["source"],"relevance":c["relevance"]} for c in chunks]}

# ---------------- Agent 7: Carbon ----------------
def run_carbon_agent(ctx, plantation, trees_to_plant=None, open_space_ha=None):
    """Carbon sequestration agent.
    trees_to_plant: explicit override (overrides open_space_ha and density default).
    open_space_ha:  total surveyed open land from OSM → trees = ha × trees_per_hectare.
    Falls back to density-class defaults when neither is supplied."""
    sp = plantation["species"][:10]
    if trees_to_plant is None:
        if open_space_ha is not None and open_space_ha > 0:
            trees_to_plant = int(open_space_ha * plantation["trees_per_hectare"])
        else:
            trees_to_plant = {"Rural": 50000, "Peri-urban": 25000,
                              "Urban": 15000, "Dense Urban": 10000}[ctx["density_class"]]
    # Sanity bounds — cap per density class; floor at 500 to keep maths meaningful
    dc_cap = {"Rural": 2_000_000, "Peri-urban": 500_000,
              "Urban": 200_000, "Dense Urban": 100_000}[ctx["density_class"]]
    trees_to_plant = max(500, min(trees_to_plant, dc_cap))
    mean_rate = float(np.mean([s["co2_kg_yr"] for s in sp]))  # kg/tree/yr mature
    def cum(years):
        tot = 0.0
        for y in range(1, years+1):
            ramp = min(0.10 + 0.10*y, 1.0)         # logistic-ish growth ramp
            surv = 0.85 if y >= 3 else (1 - 0.05*y)  # 15% mortality over first 3 yrs
            tot += trees_to_plant * mean_rate * ramp * surv
        return tot/1000.0                           # tonnes
    proj = {f"year_{y}": round(cum(y),1) for y in (1,5,10,25)}
    annual_at_maturity = trees_to_plant * mean_rate * 0.85 / 1000
    chunks = retrieve(build_retrieval_query("carbon", ctx), 2)
    return {"trees_modelled": trees_to_plant,
        "mean_species_rate_kg_yr": round(mean_rate,1),
        "cumulative_tonnes_co2e": proj,
        "annual_tonnes_at_maturity": round(annual_at_maturity,1),
        "cars_equivalent_at_maturity": int(annual_at_maturity/4.6),
        "households_equivalent": int(annual_at_maturity/1.5),
        "mortality_assumption": "15% cumulative in first 3 years (assumed)",
        "growth_curve": "saplings ~10% of mature uptake yr-1, full by ~yr-9 (IPCC Tier-1 style)",
        "confidence": "High (species-level i-Tree rates for "
                      f"{len(sp)} recommended species)",
        "grounding": [{"id":c["id"],"source":c["source"],"relevance":c["relevance"]} for c in chunks]}

# ---------------- Soil (folded into Water & Soil per architecture) ----------------
def run_soil_recommendations(ctx):
    kz = ctx["location"]["climate_zone"]
    chunks = retrieve(build_retrieval_query("soil", ctx), 2)
    recs = ["Compost 5-10 t/ha/yr; vermicompost for urban beds",
            "Biochar 2-5 t/ha (+15-25% water holding)",
            "Planting pits 1×1×1 m: 60% soil + 30% compost + 10% sand, mycorrhizal inoculation"]
    if kz.startswith("A"):       # tropical
        recs = ["Green manure (Sesbania 45-day cycle: +60-80 kg N/ha)"] + recs + \
               ["Mulching: 30-50% less surface evaporation; no bare soil in monsoon"]
    elif kz in ("BWh","BSh"):    # arid
        recs += ["Drip-basin micro-catchments around every sapling; gypsum for sodic patches",
                 "Thick mulch is critical: 50%+ evaporation cut in arid heat"]
    else:                        # temperate/continental
        recs += ["Leaf-mould composting of autumn litter; avoid winter soil compaction",
                 "Street-tree pits ≥6 m³ with structural soil under pavements",
                 "Cover crops (clover/vetch) on any soil bare over winter"]
    # Coastal / estuarine salinity — detect by high humidity + tropical/monsoon + rainfall>1500
    lat_ = ctx["location"].get("lat", 0)
    _cn_ = fetch_climate_normals(ctx["location"]["lat"], ctx["location"]["lon"]) or {}
    rain_ = _cn_.get("annual_rain_mm") or get_annual_rainfall(ctx["location"]["lat"], ctx["location"]["lon"])
    near_coast = (kz in ("Af","Am","Aw") and ctx["humidity_avg"] > 75 and rain_ > 1500
                  and abs(lat_) < 25)   # tropical coastline heuristic
    if near_coast:
        recs.append("Coastal/estuarine influence likely: prefer salt-tolerant Casuarina, Pongamia pinnata, "
                    "Heritiera littoralis; test soil EC before food crops; avoid waterlogging-sensitive species")
    return {"recommendations": recs,
            "grounding": [{"id":c["id"],"source":c["source"],"relevance":c["relevance"]} for c in chunks]}

# ---------------- Agent 9: Renewable Energy & Interventions (feasibility-checked) ----------------
GRID_EF = {"india":0.71,"bangladesh":0.67,"kenya":0.10,"nigeria":0.52,"united kingdom":0.21,
           "germany":0.38,"france":0.06,"united states":0.37,"uae":0.49,"japan":0.46,
           "brazil":0.10,"singapore":0.41,"egypt":0.47,"russia":0.36,"australia":0.66}

def run_energy_agent(ctx):
    """Every intervention is gated on SITE FEASIBILITY: terrain slope (real
    elevation data), land availability by density class, measured insolation,
    wind speed, rainfall. Never proposes clearing green cover or building on
    steep/mountainous ground."""
    L = ctx["location"]
    terrain = get_terrain(L["lat"], L["lon"]) or {
        "elevation_m": None, "relief_m_per_km": None,
        "slope_class": "unknown (offline — verify slope on site before any ground works)"}
    solar, wind = ctx.get("solar_kwh_m2_day", 4.5), ctx.get("wind_ms", 3.5)
    dens, dc = ctx["population_density"], ctx["density_class"]
    grid_ef = GRID_EF.get((L.get("country") or "").lower(), 0.44)
    kwh_kwp = solar * 0.75 * 365                       # per kWp/yr at PR 0.75
    rain = get_annual_rainfall(L["lat"], L["lon"])
    steep = terrain["slope_class"] in ("steep", "mountainous")
    items = []
    add = lambda n, f, r, p="": items.append(
        {"intervention": n, "feasible": f, "reason": r, "potential": p})

    add("Rooftop solar PV",
        "✅ Yes" if solar >= 3.5 else "⚠️ Conditional" if solar >= 2.2 else "❌ No",
        f"insolation {solar:.1f} kWh/m²/day; uses existing roofs — no land/terrain constraint",
        f"1 kWp ≈ {kwh_kwp:,.0f} kWh/yr ≈ {kwh_kwp*grid_ef/1000:.2f} t CO2/yr avoided")
    if steep:
        add("Ground-mounted solar farm", "❌ No",
            f"terrain is {terrain['slope_class']} (relief {terrain['relief_m_per_km']} m/km) — "
            "grading risks erosion/landslides; rooftop & canopy solar only")
    elif dc in ("Dense Urban", "Urban"):
        add("Ground-mounted solar farm", "⚠️ Conditional",
            "open urban land is scarce and better used for green cover — restrict to parking "
            "canopies, canal-top arrays and capped landfills",
            "canal-top: ~1 MWp per km of 10 m-wide canal + evaporation savings")
    else:
        add("Ground-mounted solar farm", "✅ Yes" if solar >= 4 else "⚠️ Conditional",
            f"{terrain['slope_class']} terrain, {dc.lower()} land — ONLY degraded/barren plots; "
            "NEVER clear vegetation for panels (carbon payback turns negative)",
            f"1 ha ≈ 0.8 MWp ≈ {0.8*kwh_kwp:,.0f} MWh/yr")
    add("Agrivoltaics (panels over crops)",
        "❌ No" if steep else "✅ Yes" if dc in ("Rural","Peri-urban") and solar >= 4 else "⚠️ Conditional",
        "terrain too steep for arrays" if steep else
        "dual land use: farmland stays productive, panels cut crop heat stress 1-3°C")
    add("Small wind turbines",
        "✅ Yes" if wind >= 5.5 else "⚠️ Conditional" if wind >= 4.0 else "❌ No",
        f"mean wind {wind:.1f} m/s here (viable ≥4, good ≥5.5)")
    if (terrain["relief_m_per_km"] or 0) >= 60 and rain >= 1000:
        add("Micro-hydro", "⚠️ Conditional",
            f"relief {terrain['relief_m_per_km']} m/km + {rain} mm rain/yr — "
            "survey perennial streams for head & flow before committing")
    else:
        add("Micro-hydro", "❌ No",
            f"insufficient slope ({terrain['relief_m_per_km'] or 'unknown'} m/km) "
            f"and/or rainfall ({rain} mm) for usable head/flow")
    add("Community biogas (organic waste)",
        "✅ Yes" if dens > 1000 else "⚠️ Conditional",
        f"{dc}: ≈{int(dens*0.35):,} kg organic waste/km²/day (0.35 kg/person) feeds digesters",
        "40-60 m³ biogas per tonne wet waste; digestate fertilises the plantations")
    if dc == "Rural":
        add("Solar irrigation pumps", "✅ Yes" if solar >= 4 else "⚠️ Conditional",
            f"replaces diesel pumps; insolation {solar:.1f} kWh/m²/day",
            "each 5 HP pump ≈ 2-3 t CO2/yr avoided")
    else:
        add("EV charging network", "✅ Yes",
            "pairs with the EV-transition zone in pollution actions; feed from rooftop solar")
    if ctx["heat_index_avg"] - ctx["temperature_avg"] > 3:
        add("Cool roofs (albedo >0.65)", "✅ Yes",
            f"heat island (+{ctx['heat_index_avg']-ctx['temperature_avg']:.1f}°C feels-like) — "
            "roof surface −25°C, AC load −15-30%")
    # ---- concrete SOLAR ACTION PLAN (aggregate rooftop math for this density) ----
    roof_cover = {"Dense Urban":0.25,"Urban":0.18,"Peri-urban":0.10,"Rural":0.04}[dc]
    pv_m2_km2 = 1e6 * roof_cover * 0.30        # 30% of roof area is PV-usable
    mwp_km2 = pv_m2_km2 / 7 / 1000             # 7 m² per kWp
    gen_gwh_km2 = mwp_km2 * 1000 * kwh_kwp / 1e6
    solar_plan = {
        "usable_rooftop_m2_per_km2": int(pv_m2_km2),
        "potential_mwp_per_km2": round(mwp_km2, 1),
        "annual_generation_gwh_per_km2": round(gen_gwh_km2, 2),
        "co2_avoided_t_per_km2_yr": int(gen_gwh_km2 * 1e6 * grid_ef / 1000),
        "phased_rollout": [
            "Year 1: public buildings + schools (5% of potential) — visible anchor projects, "
            "feed EV chargers & streetlights",
            "Years 2-3: net-metering drive for commercial roofs + housing societies (to 25%)",
            "Years 4-5: residential mass adoption with subsidy/financing linkage (to 60%)"],
        "siting_rule": ("desert advantage: highest insolation class — pair arrays with dust-"
                        "cleaning schedule and shade-giving mounting over parking/courtyards"
                        if steep is False and solar >= 5.5 else
                        "rooftops & canopies first; ground-mount ONLY on verified-flat degraded land")}
    chunks = retrieve("renewable energy siting feasibility solar slope terrain wind biogas", 2)
    return {"terrain": terrain, "solar_kwh_m2_day": round(solar,2), "wind_ms": round(wind,1),
            "grid_emission_factor_kg_kwh": grid_ef, "interventions": items,
            "solar_plan": solar_plan,
            "grounding": [{"id":c["id"],"source":c["source"],"relevance":c["relevance"]} for c in chunks]}

print("Agents defined ✓  (data, rag, pollution, plantation, water, urban, carbon, soil, energy + synthesis next)")


# ════════ 09_orchestrator.py ════════
# ============================================================
# Cell 9: Synthesis Agent + Orchestrator
#
# Entry points:
#   ask_ecogpt(query, user_type, polish, survey_radius_km)
#       → full pipeline: location → 8 agents → synthesis → optional LLM polish
#       → returns Markdown report string
#       → caches last result in ask_ecogpt.last{} for the dashboard / Streamlit
#
#   ask_followup(question)
#       → hybrid RAG + LLM (or deterministic fallback) for follow-up questions
#       → must be called after ask_ecogpt() to have context
#
#   ask_ecogpt_adk(query)  [async]
#       → same pipeline routed through Google ADK SequentialAgent when installed
#
# Design principles:
#   - Every agent runs deterministically first (scientific core always produces output)
#   - LLM (AMD vLLM / Ollama / HF) only polishes text — it never originates figures
#   - user_type controls audience-specific section appended to report
#     ("government" → policy brief + KPIs, "ngo" → community actions,
#      "researcher" → full data provenance, "default" → no extra section)
#   - All assumptions and estimates are passed through to the final report
# ============================================================

_LOC_NOISE = {"location","improve","environment","focus","water","tree","trees","pollution",
              "population","density","analysis","report","high","given","bodies","planting",
              "reduction","the","and","here","how","what","which","restoration","riverbank"}

def _location_candidates(query: str) -> list:
    """Possible place-name phrases, most explicit first."""
    cands = []
    m = re.search(r"location\s*[:\-]\s*([^\n(.;]+)", query, re.I)
    if m: cands.append(m.group(1))
    for m in re.finditer(r"\b(?:in|at|near|around|for)\s+([A-Z][\w'\-]+(?:[ ,]+[A-Z][\w'\-]+){0,3})", query):
        cands.append(m.group(1))
    q = query.strip()
    if len(q) <= 45 and "?" not in q:        # bare input like "Paris" / "Cape Town, SA"
        cands.append(q)
    for m in re.finditer(r"\b([A-Z][a-zA-Z'\-]{2,}(?:\s+[A-Z][a-zA-Z'\-]{2,}){0,3})\b", query):
        cands.append(m.group(1))
    seen, out = set(), []
    for c in cands:
        c = c.strip(" ,.;:-")
        if not c or c.lower() in seen: continue
        words = [w for w in re.findall(r"[A-Za-z'\-]+", c) if w.lower() not in _LOC_NOISE]
        if not words: continue
        seen.add(c.lower()); out.append(" ".join(words) if len(words) < len(c.split()) else c)
    return out

def parse_location(query: str):
    """Extract location → (lat, lon, label) or None. NEVER silently defaults:
    coords → gazetteer → Nominatim (any place on Earth)."""
    m = (re.search(r"lat[:\s]*(-?\d+\.?\d*)[,\s]+lon[g]?[:\s]*(-?\d+\.?\d*)", query, re.I)
         or re.search(r"\(?\s*(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)\s*\)?", query))
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        return lat, lon, f"({lat:.3f}, {lon:.3f})"
    ql = query.lower()
    for name in GAZETTEER:
        if re.search(rf"\b{re.escape(name)}\b", ql):
            return GAZETTEER[name][0], GAZETTEER[name][1], name.title()
    for cand in _location_candidates(query):
        key = cand.lower()
        for name in GAZETTEER:
            if name in key:
                return GAZETTEER[name][0], GAZETTEER[name][1], name.title()
        if CAPS["geopy"]:
            try:
                loc = Nominatim(user_agent="ecogpt_hackathon", timeout=6).geocode(cand)
                if loc:
                    return loc.latitude, loc.longitude, cand
            except Exception:
                continue
    return None

LOCATION_HELP = ("❓ **I couldn't identify a location in your message.** Please give me a city "
    "(e.g. `Nairobi`, `Location: Pune, India`) or coordinates (`22.557, 88.494`). "
    "Online geocoding covers any place on Earth; offline I know: "
    + ", ".join(sorted(n.title() for n in GAZETTEER)) + ".")

def _user_type_note(user_type: str, ctx, pol, plant, water, energy, carbon, urban) -> str:
    """Audience-specific section appended to the deterministic report."""
    if user_type == "government":
        aqi = ctx["aqi"]["score"]
        rwh = water["rainwater_harvesting"]
        return (
            "\n---\n## 🏛 Policy Brief (Government/Municipal Authority)\n"
            "*For municipal officers and regulators — enforcement, budget, and KPIs.*\n\n"
            f"**Compliance status**: AQI {aqi} — "
            f"{'⚠️ exceeds action threshold (>100); NAAQS/local standards require documented response' if aqi > 100 else '✅ within acceptable limits; preventive monitoring sufficient'}.\n"
            f"**Budget anchors** (estimated):\n"
            f"- Miyawaki micro-forest: ₹5–8L per 100-tree plot (3-yr canopy closure)\n"
            f"- Avenue planting: ₹2,000–4,000/tree installed (including pit, guard, 2-yr maintenance)\n"
            f"- RWH retrofit mandate on plots >300 m²: {rwh['per_100m2_roof_litres_yr']:,} L/yr per 100 m² (positive ROI in 3–5 yr)\n"
            f"- Rooftop solar on public buildings: {energy['solar_plan']['potential_mwp_per_km2']} MWp/km² potential\n"
            f"**KPI targets**: AQI <100 (3-yr); +{urban['required_green_ha_per_km2']} ha/km² green cover; RWH mandate enforced; "
            f"{energy['solar_plan']['co2_avoided_t_per_km2_yr']:,} t CO2 avoided/yr/km² from solar.\n"
            f"**Regulatory levers**: Tree-felling NOC reform; building by-law green-coverage mandate (30% for new construction); "
            f"vehicle emission inspection zone; industrial stack CEM (continuous emission monitoring) mandate.\n"
            f"**Carbon credits**: {carbon['annual_tonnes_at_maturity']:,} t CO2e/yr at maturity — "
            "registerable under ICCM/Gold Standard voluntary markets (consult carbon registry).\n"
        )
    elif user_type == "ngo":
        sp3 = ", ".join(s["common"] for s in plant["species"][:3])
        rwh = water["rainwater_harvesting"]
        return (
            "\n---\n## 🤝 Community Action Guide (NGO / Civil Society)\n"
            "*Low-cost, community-driven steps for field organizers and volunteers.*\n\n"
            f"**This weekend** — community nursery: grow {sp3} from seeds/cuttings "
            "(₹5–30/sapling vs ₹200+ bought); recruit one champion per street/block.\n"
            f"**Schools programme**: Each class 'adopts' 10 trees, measures growth termly — "
            "builds a local constituency and feeds data back to EcoGPT.\n"
            f"**Water body drives**: Monthly Eichhornia (water hyacinth) removal + litter days; "
            "link removed biomass to local biogas unit.\n"
            f"**Quick water win**: RWH for 10 community buildings = {rwh['per_100m2_roof_litres_yr']*10:,} L/yr saved "
            f"(10 × 100 m² roofs at {water['annual_rainfall_mm']} mm rain/yr).\n"
            f"**Advocacy asks from municipal body**: Publicly visible AQI monitors; "
            "open-plot registry protected from encroachment; RWH mandate enforced; "
            "community nurseries counted in green-cover metrics.\n"
            "**Funding signals**: CAMPA funds (India); Green Climate Fund sub-grants; CSR (Companies Act 2% mandate); "
            "MGNREGA for rural soil/water works.\n"
        )
    elif user_type == "researcher":
        return (
            "\n---\n## 🔬 Methodological Notes (Researcher)\n"
            "*Full data provenance, confidence levels, and model assumptions.*\n\n"
            f"**Data source**: {ctx['data_source']}\n"
            f"**Sensor completeness**: {ctx['data_completeness']:.0%} | Readings used: {ctx['readings_used']:,}\n"
            f"**AQI model**: EPA breakpoint linear interpolation; sensor rescaling applied: "
            f"{'; '.join(ctx['aqi'].get('notes', [])) or 'none'}.\n"
            f"**Climate data**: ERA5 12-month trailing archive for exact coordinates (not nearest-city lookup); "
            f"Köppen class computed from 12 monthly temp + rain normals via _koppen_from_monthlies().\n"
            f"**Carbon model**: IPCC Tier-1-style logistic growth ramp (min(0.10+0.10y, 1.0)); "
            f"15% cumulative mortality first 3 yr; i-Tree species-level CO2 rates; {carbon['confidence']}.\n"
            f"**OSM land survey**: Overpass API, {ctx.get('survey_radius_km', 3)} km radius; "
            f"tree capacity 1 tree/25 m² (open land); polygon area by Shoelace/equirectangular approx.\n"
            f"**Terrain**: Open-Meteo elevation API 9-point grid (~1 km spacing) → slope class.\n"
            f"**Population density**: OSM place-node population attribute → gazetteer → rural 300/km² fallback.\n"
            f"**Assumptions flagged**: {'; '.join(ctx['assumptions']) or 'none'}.\n"
            f"**Confidence summary**: Climate — High (ERA5 real data); Carbon — {carbon['confidence']}; "
            f"AQI — {'High (IoT sensors)' if 'IoT' in ctx['data_source'] else 'Medium (modelled/CAMS)'}; "
            f"Population — {'Medium (OSM)' if ctx['population_density'] > 200 else 'Low (rural estimate)'}.\n"
        )
    return ""  # "default" — no extra section


def run_synthesis(ctx, pol, plant, water, urban, carbon, soil, energy, user_type="default"):
    """Assemble the final Markdown report from all 8 agent outputs.

    Deterministic — no LLM call here. The LLM polish step is done in ask_ecogpt()
    after this function returns, so the raw report is always available as fallback.

    Sections produced:
      At a Glance table (letter grades) → Environmental Assessment → Key Risks →
      Recommended Species table → Plantation Plan → Pollution Actions →
      Soil → Water → Biodiversity → Carbon table → Energy plan →
      What-If Simulation → Priority Action Items (location-specific, max 10) →
      Audience-specific section (_user_type_note)

    Args:
        ctx    : Data ingestion output (AQI, temp, humidity, location, assumptions …)
        pol    : Pollution agent output (source apportionment, actions, bio-filter species)
        plant  : Plantation agent output (species list, density, zones, canopy layers)
        water  : Water agent output (stress level, restoration plan, RWH figures)
        urban  : Urban agent output (green-space deficit, density class, toolkit)
        carbon : Carbon agent output (sequestration table, cars-equivalent, confidence)
        soil   : Soil agent output (recommendations list)
        energy : Energy agent output (solar plan, interventions, terrain)
        user_type: "government" | "ngo" | "researcher" | "default"

    Returns:
        Markdown string (stripped, ready for display or LLM refinement).
    """
    L = ctx["location"]
    sp_rows = "\n".join(
        f"| {s['scientific']} | {s['common']} | {s['role'][:45]} | {s['co2_kg_yr']} | {s['water']} | {s['native_to']} |"
        for s in plant["species"])
    pr = carbon["cumulative_tonnes_co2e"]
    # scenarios anchored to actual trees_modelled (from real OSM open-space data when available)
    base_n = carbon["trees_modelled"]   # dynamic — based on surveyed open land, not a hardcoded constant
    area_ha = {"Rural": 150000, "Peri-urban": 60000, "Urban": 35000, "Dense Urban": 20000}[ctx["density_class"]]
    scenario_set = [(base_n, 5), (base_n*3, 10), (base_n*6, 25)]
    sims = [simulate_intervention(ctx["aqi"]["score"], n, n/plant["trees_per_hectare"], y,
                                  urban_area_ha=area_ha) for n, y in scenario_set]
    sim_txt = "\n".join(
        f"- **If {n:,} trees are planted over {y} years:** AQI {ctx['aqi']['score']} → "
        f"{s['projected_aqi']} (−{s['aqi_reduction']}), {s['co2_captured_tonnes']:,} t CO2e captured "
        f"(≈{s['cars_equivalent']:,} cars/yr), +{s['green_cover_increase_pct']}% green cover *(medium confidence)*"
        for (n, y), s in zip(scenario_set, sims))
    # biodiversity narrative from the ACTUAL species mix + climate zone
    kz = L["climate_zone"]
    bloom = [f"{s['common']} ({s['season']})" for s in plant["species"][:6]]
    keystone = next((s for s in plant["species"] if "keystone" in s["role"].lower()
                     or "Ficus" in s["scientific"] or "Quercus" in s["scientific"]), None)
    bio_lines = [
        f"- 5-layer canopy from the selected mix (see Plantation Plan); planting windows: {', '.join(bloom)}",
        (f"- Keystone species: *{keystone['scientific']}* ({keystone['common']}) — anchor for birds & pollinators"
         if keystone else "- Add a keystone fig/oak equivalent for this zone to anchor vertebrate food webs"),
        "- Stepping-stone pocket parks every 500 m; dead-wood retention; no-mow meadow patches",
        ("- Wetland margins <30 cm deep + reed beds for waterbirds" if kz.startswith("A") else
         "- Flowering desert natives + shaded water points for arid-zone pollinators and birds"
         if kz.startswith("B") else
         "- Hedgerow connectivity + nectar strips for temperate pollinators")]
    anomalies = "\n".join(f"- {a}" for a in ctx["anomalies"]) or "- None flagged"
    # ---- priority list assembled ONLY from this location's actual problems ----
    _top_energy = next((i for i in energy["interventions"] if i["feasible"].startswith("✅")),
                       energy["interventions"][0])
    prio_items = [water["restoration_plan"][0]]
    if ctx["aqi"]["score"] > 100:
        prio_items += [pol["actions"]["short_term_24_72h"][0], pol["actions"]["medium_term_1_6mo"][0]]
    else:
        prio_items.append("Air already within acceptable limits — protect it: keep monitoring, "
                          "enforce against new emission sources (no emergency action needed)")
    prio_items.append(f"Launch {plant['species'][0]['season']} plantation: "
                      f"{plant['trees_per_hectare']} trees/ha mixed natives in: {plant['priority_zones'][0]}")
    prio_items.append(f"{_top_energy['intervention']} — {_top_energy['reason'][:90]}")
    prio_items.append(urban["dense_urban_toolkit"][0] if ctx["density_class"] == "Dense Urban"
                      else urban["greening_without_displacement"][0] + " greening drive")
    if (not water["stress_level"].startswith("Low")) or L["climate_zone"].startswith("B"):
        prio_items.append(f"Rainwater harvesting: "
                          f"{water['rainwater_harvesting']['per_100m2_roof_litres_yr']:,} L/yr "
                          "per 100 m² roof — mandate + retrofit programme")
    prio_items.append(soil["recommendations"][0])
    if ctx["aqi"]["score"] > 100:
        prio_items.append(pol["actions"]["long_term_1_5yr"][0])
    if pol["heat_island_delta"] > 3:
        prio_items.append("Heat-island programme: cool roofs (albedo >0.65) + 30% canopy target")
    prio_items.append(f"Biodiversity: {bio_lines[1].lstrip('- ')}")
    prio_items.append("Quarterly review vs this baseline (re-run EcoGPT)")
    prio_items = prio_items[:10]
    # ---------- plain-language "At a Glance" with letter grades ----------
    def _grade(val, bands):  # bands = [(ceiling, grade), …]
        for ceil_, gr in bands:
            if val <= ceil_: return gr
        return "F"
    # ---- condition flags: healthy domains get "no major change required" treatment ----
    arid = L["climate_zone"].startswith("B")
    ok_air = ctx["aqi"]["score"] <= 100
    ok_heat = pol["heat_island_delta"] <= 3
    ok_water = water["stress_level"].startswith("Low")
    air_status = (f"✅ **Air quality here is already {ctx['aqi']['category']}** — no emergency "
                  "measures required; the actions below are protective maintenance only.\n"
                  if ok_air else "")
    water_status = ("✅ **Water situation is comfortable** — major new water works may not be "
                    "required; prioritise protection of existing bodies and drainage.\n"
                    if ok_water else "")
    green_status = ("ℹ️ **Low population density — existing green cover is likely adequate.** "
                    "Focus on PROTECTING habitat and restoration quality rather than chasing "
                    "new planting targets.\n" if ctx["density_class"] == "Rural" and not arid else "")
    desert_note = (f"\n🏜 **Desert climate ({water['annual_rainfall_mm']} mm/yr):** every "
                   "recommendation below follows water-budget-first logic — reduced planting "
                   "density with micro-catchments, harvest-and-store water strategy, dust-control "
                   "shelterbelts, and solar as the highest-leverage intervention.\n" if arid else "")
    g_air = _grade(ctx["aqi"]["score"], [(50,"A"),(100,"B"),(150,"C"),(200,"D"),(300,"E")])
    g_heat = _grade(pol["heat_island_delta"], [(1,"A"),(3,"B"),(5,"C"),(8,"D")])
    g_water = {"L":"B","M":"B","H":"D"}.get(water["stress_level"][0], "C")
    sp3 = ", ".join(s["common"] for s in plant["species"][:3])
    y10 = carbon["cumulative_tonnes_co2e"]["year_10"]
    at_a_glance = f"""## 📌 At a Glance (easy summary)
| | Status | Grade |
|---|---|---|
| 🌬 Air | AQI **{ctx['aqi']['score']} – {ctx['aqi']['category']}**, mainly from **{pol['source_apportionment']['classification']}** sources | **{g_air}** |
| 🔥 Heat | Feels **{pol['heat_island_delta']}°C hotter** than actual ({"heat-island problem" if pol['heat_island_flag'] else "acceptable"}) | **{g_heat}** |
| 💧 Water | Stress: **{water['stress_level'].split('(')[0].strip()}** · rain ≈{water['annual_rainfall_mm']} mm/yr | **{g_water}** |
| 🌳 Green | **{ctx['density_class']}** ({ctx['population_density']:,}/km²) → needs ~**{urban['required_green_ha_per_km2']} ha/km²** more green | **{"D" if ctx['density_class']=="Dense Urban" else "C" if ctx['density_class']=="Urban" else "B"}** |
| ⚡ Energy | Solar **{energy['solar_kwh_m2_day']} kWh/m²/day** · wind {energy['wind_ms']} m/s · terrain **{energy['terrain']['slope_class']}** | **{_grade(-energy['solar_kwh_m2_day'], [(-5.5,"A"),(-4.5,"B"),(-3.5,"C"),(-2.5,"D")])}** |

**In one paragraph:** {f"{L['city']}'s air is already *{ctx['aqi']['category'].lower()}* — protect it and put energy into the bigger levers here" if ok_air else f"{L['city']}'s air is *{ctx['aqi']['category'].lower()}* and the biggest quick win is tackling {pol['source_apportionment']['classification']} emissions"}. {"Plant water-wise natives only (" + sp3 + f") at reduced density with micro-catchment basins — the {water['annual_rainfall_mm']} mm water budget rules out dense forestry, and rooftop solar (" + str(energy['solar_kwh_m2_day']) + " kWh/m²/day) is this region's strongest intervention" if arid else f"Plant **{carbon['trees_modelled']:,} trees** (start with {sp3}) on the open land shown in the dashboard — in 10 years that captures ≈**{y10:,.0f} t of CO2** (like taking {int(y10/4.6/10):,} cars off the road each year)"}. {"Store every drop: " + water['restoration_plan'][1].lower() if arid else f"Harvest rooftop rain (**{water['rainwater_harvesting']['per_100m2_roof_litres_yr']:,} L/yr per 100 m² roof**) and start water work with: *{water['restoration_plan'][0].lower()}*"}.
{desert_note}"""
    all_assumptions = ctx["assumptions"] + ctx["aqi"].get("notes", [])
    assumptions = "; ".join(all_assumptions) if all_assumptions else "none"
    report = f"""
# 🌿 EcoGPT Environmental Report — {L['city']}, {L['country']}

{at_a_glance}

## Environmental Assessment
**Location:** {L['city']} ({L['lat']:.3f}, {L['lon']:.3f}) | **Climate:** {L['climate_name']} ({L['climate_zone']})
**AQI:** {ctx['aqi']['score']} ({ctx['aqi']['category']}) — dominant pollutant: {ctx['aqi']['dominant']}
**Temp:** {ctx['temperature_avg']}°C | **Humidity:** {ctx['humidity_avg']}% | **Heat Index:** {ctx['heat_index_avg']}°C
**Population density:** ~{ctx['population_density']:,}/km² ({ctx['density_class']})
**Data source:** {ctx['data_source']} | **Readings:** {ctx['readings_used']:,} (completeness {ctx['data_completeness']:.0%})
**Assumptions:** {assumptions}

## Key Risks Identified
- Pollution severity **{pol['severity']}**; likely source mix: **{pol['source_apportionment']['classification']}** (signals {pol['source_apportionment']['signals']})
- {"⚠️ **Heat island**: heat index exceeds ambient by " + str(pol['heat_island_delta']) + "°C" if pol['heat_island_flag'] else "Heat island risk currently low (HI−TMP = " + str(pol['heat_island_delta']) + "°C)"}
- Water stress: **{water['stress_level']}** (annual rainfall {water['annual_rainfall_mm']} mm — {water['rainfall_source']})
- Green-space requirement: **{urban['required_green_ha_per_km2']} ha/km²** to meet WHO 9 m²/capita ({urban['note']})
- Sensor anomalies:
{anomalies}

## Recommended Plant Species
| Scientific name | Common | Role | CO2 (kg/yr) | Water | Status |
|---|---|---|---|---|---|
{sp_rows}

*Avoid (invasive):* {", ".join(plant['avoid'])}. *Rule:* {plant['polyculture_rule']}.

## Plantation Plan
{green_status}- **Density:** {plant['trees_per_hectare']} trees/ha ({ctx['density_class']}{", water-budget-capped for desert" if arid else ""} scaling)
- **Arrangement:** {plant['spatial_arrangement']}
- **Priority zones:** {"; ".join(plant['priority_zones'])}
- **Canopy layers (from selected species):** {"; ".join(f"{lyr} — {names}" for lyr, names in plant['canopy_layers'].items() if not names.startswith("—"))}

## Pollution Reduction Actions
{air_status}**Short-term (24–72 h):** {"; ".join(pol['actions']['short_term_24_72h'])}
**Medium-term (1–6 mo):** {"; ".join(pol['actions']['medium_term_1_6mo'])}
**Long-term (1–5 yr):** {"; ".join(pol['actions']['long_term_1_5yr'])}
**Bio-filters:** NO2 → {", ".join(pol['biofilter_species']['NO2'])} | PM → {", ".join(pol['biofilter_species']['PM2.5/dust'])} | VOC → {", ".join(pol['biofilter_species']['VOC'])}
**Expected gain:** {pol['expected_improvement']}

## Soil Improvement Actions
{chr(10).join("- " + r for r in soil['recommendations'])}

## Water Conservation Recommendations
{water_status}{chr(10).join("- " + r for r in water['restoration_plan'])}
- **RWH:** {water['rainwater_harvesting']['per_100m2_roof_litres_yr']:,} L/yr per 100 m² roof ({water['rainwater_harvesting']['household_demand_coverage_pct']}% of a 5-person household demand); {water['rainwater_harvesting']['recharge_structures']}
- **Groundwater:** {water['rainwater_harvesting']['groundwater_response']}
- **Irrigation:** {"; ".join(water['irrigation'])}

## Biodiversity Enhancement Plan
{chr(10).join(bio_lines)}

## Carbon Sequestration Estimate ({carbon['trees_modelled']:,} trees, mean {carbon['mean_species_rate_kg_yr']} kg CO2/tree/yr)
| Year 1 | Year 5 | Year 10 | Year 25 |
|---|---|---|---|
| {pr['year_1']:,} t | {pr['year_5']:,} t | {pr['year_10']:,} t | {pr['year_25']:,} t |

At maturity: **{carbon['annual_tonnes_at_maturity']:,} t CO2e/yr** ≈ {carbon['cars_equivalent_at_maturity']:,} cars removed ≈ {carbon['households_equivalent']:,} households' electricity. {carbon['mortality_assumption']}; {carbon['growth_curve']}. Confidence: {carbon['confidence']}.
{urban['co2_honesty_note']} *(estimated)*

## Renewable Energy & Other Improvement Scopes (feasibility-checked for this site)
**Site:** terrain **{energy['terrain']['slope_class']}**{f" (elev {energy['terrain']['elevation_m']} m, relief {energy['terrain']['relief_m_per_km']} m/km)" if energy['terrain']['elevation_m'] is not None else ""} | **Solar:** {energy['solar_kwh_m2_day']} kWh/m²/day | **Wind:** {energy['wind_ms']} m/s | **Grid:** {energy['grid_emission_factor_kg_kwh']} kg CO2/kWh

| Intervention | Feasible here? | Site-specific reason | Potential |
|---|---|---|---|
{chr(10).join(f"| {i['intervention']} | {i['feasible']} | {i['reason']} | {i['potential']} |" for i in energy['interventions'])}

**☀️ Solar Action Plan ({ctx['density_class']}):** usable rooftop ≈{energy['solar_plan']['usable_rooftop_m2_per_km2']:,} m²/km² → **{energy['solar_plan']['potential_mwp_per_km2']} MWp/km²** generating {energy['solar_plan']['annual_generation_gwh_per_km2']} GWh/yr/km² and avoiding **{energy['solar_plan']['co2_avoided_t_per_km2_yr']:,} t CO2/yr/km²**. Rollout: {" → ".join(energy['solar_plan']['phased_rollout'])}. Siting: {energy['solar_plan']['siting_rule']}.

## Predicted Environmental Impact
With full implementation over 5 years: AQI improvement {"15–30" if ctx['aqi']['score'] > 150 else "5–15"} points locally *(estimated)*{", ambient cooling 1–3°C in greened zones" if pol['heat_island_flag'] else ""}, runoff reduction 50–80% on treated catchments, measurable pollinator and bird recovery within 3 years.

## Priority Action Items (only what THIS location actually needs)
{chr(10).join(f"{i}. {item}" for i, item in enumerate(prio_items, 1))}

## What-If Simulation
{sim_txt}

---
*Grounded in: {", ".join(sorted(set(g['source'] for agent in (pol,plant,water,urban,carbon,soil,energy) for g in agent['grounding'])))}.*
*Figures marked (estimated)/(assumed) follow the EcoGPT no-fabrication policy.*
{_user_type_note(user_type, ctx, pol, plant, water, energy, carbon, urban)}"""
    return report.strip()

def ask_ecogpt(query: str, user_type: str = "default", polish: bool = True,
               survey_radius_km: float = 3.0) -> str:
    """Full EcoGPT pipeline: location → ingestion → specialists → synthesis → (LLM polish).
    survey_radius_km controls the OSM open-space land survey radius (default 3 km)."""
    loc = parse_location(query)
    if loc is None:
        return LOCATION_HELP
    lat, lon, label = loc
    ctx    = run_data_ingestion(lat, lon, label=label)
    ctx["survey_radius_km"] = round(survey_radius_km, 1)   # pass through for researcher notes
    pol    = run_pollution_agent(ctx)
    plant  = run_plantation_agent(ctx)
    water  = run_water_agent(ctx)
    urban  = run_urban_agent(ctx)

    # Real open-space survey → accurate tree-count for carbon modelling
    try:
        spaces = get_open_spaces(lat, lon, radius_m=int(survey_radius_km * 1000))
    except Exception:
        spaces = []
    open_plots = [s for s in spaces if s["kind"] == "open"]
    total_open_ha = sum(s["area_m2"] for s in open_plots) / 10_000.0

    carbon = run_carbon_agent(ctx, plant, open_space_ha=total_open_ha if total_open_ha > 0 else None)
    soil   = run_soil_recommendations(ctx)
    energy = run_energy_agent(ctx)
    report = run_synthesis(ctx, pol, plant, water, urban, carbon, soil, energy, user_type)
    if polish and LLM.kind != "none":
        polished = LLM.generate(
            SYSTEM_PROMPTS["synthesis"] + f"\nUser type: {user_type}. Keep ALL numbers, species names and "
            "tables EXACTLY as given. Improve flow and tone only; do not add facts.",
            f"User query: {query}\n\nDraft report to refine:\n{report}", max_tokens=1800)
        if polished and len(polished) > 400:
            report = polished
    ask_ecogpt.last = {"ctx":ctx,"pol":pol,"plant":plant,"water":water,
                       "urban":urban,"carbon":carbon,"soil":soil,"energy":energy,
                       "spaces": spaces}   # cached here — Streamlit uses this, no duplicate OSM call
    return report

def _followup_from_context(question: str, ctx: dict, last: dict) -> str:
    """Context-aware deterministic answer when no LLM is available.
    Routes to the actual pipeline data for this location."""
    q = question.lower()
    plant  = last.get("plant", {})
    water  = last.get("water", {})
    energy = last.get("energy", {})
    pol    = last.get("pol", {})
    carbon = last.get("carbon", {})
    soil   = last.get("soil", {})
    L      = ctx.get("location", {})
    city   = L.get("city", "this location")
    kz     = L.get("climate_name", L.get("climate_zone", ""))

    if any(w in q for w in ["tree","plant","species","forest","canopy","sapling","native","invasiv"]):
        sp_rows = "\n".join(
            f"- *{s['scientific']}* (**{s['common']}**) — {s['role'][:65]}, "
            f"{s['co2_kg_yr']} kg CO2/yr, water need: {s['water']}"
            for s in plant.get("species", [])[:8])
        return (f"**Species recommended for {city}** ({kz}):\n{sp_rows}\n\n"
                f"**Density**: {plant.get('trees_per_hectare')} trees/ha "
                f"({'desert-mode reduced' if L.get('climate_zone','').startswith('B') else 'standard'})\n"
                f"**Priority planting zones**: {'; '.join(plant.get('priority_zones', [])[:3])}\n"
                f"**Arrangement**: {plant.get('spatial_arrangement','')[:120]}\n"
                f"**Avoid (invasive)**: {', '.join(plant.get('avoid', []))}\n"
                f"**Polyculture rule**: {plant.get('polyculture_rule','')}")

    if any(w in q for w in ["water","rain","harvest","pond","river","flood","drought",
                             "khadin","irrigation","rwh","wetland","stream"]):
        rwh = water.get("rainwater_harvesting", {})
        plan = "\n".join(f"- {r}" for r in water.get("restoration_plan", [])[:6])
        return (f"**Water situation for {city}**: {water.get('stress_level')}\n"
                f"Annual rainfall: **{water.get('annual_rainfall_mm')} mm/yr** "
                f"({water.get('rainfall_source','estimate')})\n\n"
                f"**Plan**:\n{plan}\n\n"
                f"**Rainwater harvesting**: {rwh.get('per_100m2_roof_litres_yr',0):,} L/yr per 100 m² roof "
                f"(covers **{rwh.get('household_demand_coverage_pct',0)}%** of a 5-person household demand)\n"
                f"**Recharge**: {rwh.get('recharge_structures','')}\n"
                f"**Irrigation**: {'; '.join(water.get('irrigation', []))}")

    if any(w in q for w in ["solar","energy","wind","power","renewable","electricity","biogas",
                             "photovoltaic","pv","turbine","agrivoltaic"]):
        feasible = [i for i in energy.get("interventions", []) if "✅" in str(i.get("feasible",""))]
        cond     = [i for i in energy.get("interventions", []) if "⚠️" in str(i.get("feasible",""))]
        no_go    = [i for i in energy.get("interventions", []) if "❌" in str(i.get("feasible",""))]
        sp = energy.get("solar_plan", {})
        rows = "\n".join(f"✅ **{i['intervention']}**: {i['reason'][:90]}" for i in feasible)
        rows += "\n" + "\n".join(f"⚠️ **{i['intervention']}**: {i['reason'][:90]}" for i in cond)
        rows += "\n" + "\n".join(f"❌ **{i['intervention']}**: {i['reason'][:90]}" for i in no_go)
        return (f"**Renewable energy for {city}** — solar {energy.get('solar_kwh_m2_day')} kWh/m²/d, "
                f"wind {energy.get('wind_ms')} m/s, terrain: {energy.get('terrain',{}).get('slope_class','?')}\n\n"
                f"**Interventions (feasibility-checked)**:\n{rows.strip()}\n\n"
                f"**Solar action plan**: {sp.get('usable_rooftop_m2_per_km2',0):,} m²/km² rooftop usable → "
                f"**{sp.get('potential_mwp_per_km2')} MWp/km²** → {sp.get('annual_generation_gwh_per_km2')} GWh/yr → "
                f"**{sp.get('co2_avoided_t_per_km2_yr',0):,} t CO2 avoided/yr/km²**\n"
                f"Rollout: {' → '.join(sp.get('phased_rollout',[]))}")

    if any(w in q for w in ["aqi","air quality","pollution","no2","pm2","pm10","smog",
                             "particulate","nitro","carbon monoxide","voc","dust"]):
        actions = pol.get("actions", {})
        short_  = "; ".join(actions.get("short_term_24_72h", [])[:2])
        medium_ = "; ".join(actions.get("medium_term_1_6mo", [])[:2])
        return (f"**Air quality for {city}**: AQI **{ctx.get('aqi',{}).get('score')} "
                f"({ctx.get('aqi',{}).get('category')})**\n"
                f"Dominant pollutant: {ctx.get('aqi',{}).get('dominant','?')}\n"
                f"Source type: **{pol.get('source_apportionment',{}).get('classification','?')}** emissions "
                f"(signals: {pol.get('source_apportionment',{}).get('signals',{})})\n"
                f"Heat-island delta: {pol.get('heat_island_delta','?')}°C "
                f"({'⚠️ flagged' if pol.get('heat_island_flag') else '✅ acceptable'})\n\n"
                f"**Short-term (24–72 h)**: {short_}\n"
                f"**Medium-term (1–6 mo)**: {medium_}\n"
                f"**Expected gain**: {pol.get('expected_improvement','')}")

    if any(w in q for w in ["carbon","co2","sequestration","sequester","offset","emit","greenhouse"]):
        pr = carbon.get("cumulative_tonnes_co2e", {})
        return (f"**Carbon sequestration — {city}** "
                f"({carbon.get('trees_modelled',0):,} trees, mean {carbon.get('mean_species_rate_kg_yr',0)} kg CO2/tree/yr)\n"
                f"| Year 1 | Year 5 | Year 10 | Year 25 |\n|---|---|---|---|\n"
                f"| {pr.get('year_1',0):,} t | {pr.get('year_5',0):,} t | "
                f"{pr.get('year_10',0):,} t | {pr.get('year_25',0):,} t |\n\n"
                f"At maturity: **{carbon.get('annual_tonnes_at_maturity',0):,} t CO2e/yr** "
                f"≈ {carbon.get('cars_equivalent_at_maturity',0):,} cars removed "
                f"≈ {carbon.get('households_equivalent',0):,} households' electricity.\n"
                f"Confidence: {carbon.get('confidence','')}")

    if any(w in q for w in ["soil","organic","compost","fertilizer","biochar","mulch","nutrients",
                             "nitrogen","ph","sandy","clay"]):
        recs = "\n".join(f"- {r}" for r in soil.get("recommendations", []))
        return f"**Soil recommendations for {city}** ({kz}):\n{recs}"

    if any(w in q for w in ["population","density","urban","miyawaki","green space","greening",
                             "who","per capita","residents"]):
        u = last.get("urban", {})
        return (f"**Urban greening for {city}**: {u.get('classification','?')} "
                f"({ctx.get('population_density',0):,}/km²)\n"
                f"WHO green-space deficit: **{u.get('required_green_ha_per_km2','?')} ha/km²** to meet 9 m²/capita norm\n"
                f"**Greening options (no displacement)**: {'; '.join(u.get('greening_without_displacement',[])[:4])}\n"
                f"**Dense-urban toolkit**: {'; '.join(u.get('dense_urban_toolkit',[])[:3])}\n"
                f"**CO2 context**: {u.get('co2_honesty_note','')}")

    # Generic — give full at-a-glance snapshot
    return (f"**{city} snapshot** ({kz}, AQI {ctx.get('aqi',{}).get('score')} "
            f"{ctx.get('aqi',{}).get('category')}):\n"
            f"- 🌡 {ctx.get('temperature_avg')}°C avg, {ctx.get('humidity_avg')}% RH, "
            f"feels {ctx.get('heat_index_avg')}°C\n"
            f"- 💧 {water.get('annual_rainfall_mm')} mm/yr rain — stress: {water.get('stress_level','?')}\n"
            f"- ⚡ Solar {energy.get('solar_kwh_m2_day')} kWh/m²/d, wind {energy.get('wind_ms')} m/s\n"
            f"- 🌳 {carbon.get('trees_modelled',0):,} trees modelled → "
            f"{carbon.get('cumulative_tonnes_co2e',{}).get('year_10',0):,} t CO2 in 10 yr\n\n"
            "Ask about: **tree species · water · solar energy · air quality · carbon · soil · urban greening**")


def ask_followup(question: str) -> str:
    """Free-form Q&A: hybrid RAG retrieval + last pipeline context + LLM.
    Deterministic fallback synthesises location-aware answers without an LLM."""
    try:
        chunks = retrieve(question, 5)
    except Exception:
        chunks = []
    last = getattr(ask_ecogpt, "last", {})
    ctx  = last.get("ctx")
    evidence = "\n\n".join(
        f"[{c['source']} | rel {c['relevance']}]\n{c['text']}" for c in chunks)
    if LLM.kind != "none":
        try:
            ans = LLM.generate(
                "You are EcoGPT, an environmental advisory assistant. Answer ONLY from the evidence and "
                "context provided. Cite sources inline. If evidence is insufficient say so explicitly.",
                f"Context: {json.dumps(ctx, default=str)[:1500] if ctx else 'none yet'}\n\n"
                f"Evidence:\n{evidence}\n\nQuestion: {question}")
            if ans: return ans
        except Exception:
            pass

    # --- deterministic fallback ---
    if ctx and last:
        return _followup_from_context(question, ctx, last)

    # No location run yet — surface the best raw chunks
    good = [c for c in chunks if c.get("relevance", 0) > 0.001 and c.get("source") != "none"]
    if good:
        parts = ["**Relevant knowledge base entries:**\n"]
        for c in good[:3]:
            parts.append(f"**{c['source']}**:\n{c['text'][:600]}…\n")
        parts.append("\n*Run a location analysis first for context-specific answers. "
                     "Set HF_TOKEN or start Ollama for AI-synthesized responses.*")
        return "\n\n".join(parts)

    return ("No relevant knowledge found for that question.\n"
            "Try asking about: **tree species · water harvesting · air pollution · solar energy · "
            "soil · carbon sequestration · urban greening**.\n"
            "Run a location analysis first to get answers specific to your chosen site.")

# ---------- Google ADK wiring (used when google-adk is installed) ----------
ADK_READY = False
if CAPS["google.adk"]:
    try:
        from google.adk.agents import LlmAgent, SequentialAgent
        from google.adk.tools import FunctionTool
        from google.adk.runners import InMemoryRunner
        _model = f"ollama/{LLM.model}" if LLM.kind == "ollama" else "gemini-2.0-flash"
        _tools = {n: FunctionTool(f) for n, f in [
            ("load_sensor", load_sensor_data), ("aqi", compute_aqi),
            ("geocode", reverse_geocode), ("population", get_population_density),
            ("retrieve", retrieve), ("species", filter_species_by_climate),
            ("simulate", simulate_intervention)]}
        _agents = []
        for name, prompt, tool_keys in [
            ("data_ingestion_agent", SYSTEM_PROMPTS["data_ingestion"], ["load_sensor","aqi","geocode","population"]),
            ("rag_retrieval_agent",  SYSTEM_PROMPTS["rag_retrieval"],  ["retrieve"]),
            ("pollution_agent",      SYSTEM_PROMPTS["pollution"],      ["aqi","retrieve"]),
            ("plantation_agent",     SYSTEM_PROMPTS["plantation"],     ["species","retrieve"]),
            ("water_agent",          SYSTEM_PROMPTS["water"],          ["retrieve"]),
            ("urban_planning_agent", SYSTEM_PROMPTS["urban"],          ["population","retrieve"]),
            ("carbon_agent",         SYSTEM_PROMPTS["carbon"],         ["retrieve","simulate"]),
            ("synthesis_agent",      SYSTEM_PROMPTS["synthesis"],      [])]:
            _agents.append(LlmAgent(name=name, model=_model, instruction=prompt,
                                    tools=[_tools[k] for k in tool_keys]))
        adk_orchestrator = SequentialAgent(name="ecogpt_orchestrator", sub_agents=_agents)
        adk_runner = InMemoryRunner(agent=adk_orchestrator)
        ADK_READY = True
        print("Google ADK orchestrator wired ✓ (SequentialAgent, 8 sub-agents)")
    except Exception as e:
        print(f"ADK present but wiring failed ({e}) — built-in orchestrator in use")
else:
    print("google-adk not installed — built-in sequential orchestrator in use "
          "(identical agent prompts & tools; pip install google-adk to switch)")

async def ask_ecogpt_adk(location_query: str) -> str:
    """Run the pipeline through Google ADK (requires ADK + a live model)."""
    if not ADK_READY:
        return ask_ecogpt(location_query)
    session = await adk_runner.session_service.create_session(app_name="ecogpt", user_id="user")
    from google.genai import types as _t
    out = []
    async for ev in adk_runner.run_async(user_id="user", session_id=session.id,
            new_message=_t.Content(role="user", parts=[_t.Part(text=location_query)])):
        if ev.content and ev.content.parts:
            out += [p.text for p in ev.content.parts if getattr(p, "text", None)]
    return "\n".join(out)

print("Orchestrator ready ✓  → ask_ecogpt('Kolkata'), ask_followup('…')")

