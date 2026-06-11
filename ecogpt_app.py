"""
EcoGPT — production Streamlit UI (AMD-TCS Hackathon)
Run:  streamlit run ecogpt_app.py        (or launch from the notebook's Streamlit cell)
Needs ecogpt_core.py + enviro_sensorvalues_*.csv beside this file.
"""
import streamlit as st

st.set_page_config(page_title="EcoGPT — Environmental Intelligence", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 1.2rem;}
  div[data-testid="stMetric"] {background: rgba(46,125,50,.07); border: 1px solid rgba(46,125,50,.25);
      border-radius: 12px; padding: 10px 14px;}
  div[data-testid="stMetricLabel"] {font-size: .8rem;}
  .stTabs [data-baseweb="tab"] {font-weight: 600;}
  h1 {color: #2e7d32;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🌿 Booting the EcoGPT multi-agent engine (first run takes a minute)…")
def boot():
    import ecogpt_core as core
    return core

core = boot()

# ───────────────────────── sidebar ─────────────────────────
with st.sidebar:
    st.markdown("## 🌿 EcoGPT")
    st.caption("Multi-agent RAG environmental advisory — any location on Earth")
    query = st.text_input("📍 Location", placeholder="Kolkata · Jaisalmer · 22.557, 88.494",
                          key="loc_input")
    audience = st.selectbox("Audience", ["default", "government", "ngo", "researcher"])
    radius_km = st.slider("Land-survey radius (km)", 1, 8, 3)
    run = st.button("Run analysis 🚀", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"LLM backend: **{core.LLM.kind}**" + (f" ({core.LLM.model})" if core.LLM.model else
               " — deterministic engine"))
    st.caption(f"Retrieval: **{' + '.join(core.retriever.mode)}**")
    st.caption("Data: IoT sensors → live Open-Meteo CAMS/ERA5 → calibrated fallbacks")

if run and query.strip():
    loc = core.parse_location(query.strip())
    if loc is None:
        st.error("Couldn't identify that location — try a city name or `lat, lon` coordinates.")
    else:
        with st.spinner(f"8 agents analysing {loc[2]} …"):
            report = core.ask_ecogpt(query.strip(), user_type=audience, polish=False)
            st.session_state["report"] = report
            st.session_state["bundle"] = dict(core.ask_ecogpt.last)
            st.session_state["loc"] = loc
            st.session_state["spaces"] = core.get_open_spaces(loc[0], loc[1], radius_km * 1000)

if "bundle" not in st.session_state:
    st.title("🌿 EcoGPT — Environmental Intelligence")
    st.markdown("Enter **any city or coordinates** in the sidebar and hit *Run analysis*. "
                "EcoGPT pulls live satellite/model data for that exact spot, runs 8 specialist "
                "agents over a curated knowledge base, and returns a feasibility-checked action plan.")
    st.info("Try: `Kolkata` (IoT sensor site) · `Jaisalmer` (desert logic) · `Shimla` (mountain terrain) · `London`")
    st.stop()

b = st.session_state["bundle"]
ctx, pol, water, energy = b["ctx"], b["pol"], b["water"], b["energy"]
plant, carbon, urban = b["plant"], b["carbon"], b["urban"]
lat, lon, label = st.session_state["loc"]
spaces = st.session_state.get("spaces", [])
open_plots = [s for s in spaces if s["kind"] == "open"]
rivers = [s for s in spaces if s["kind"] == "river"]
waterbodies = [s for s in spaces if s["kind"] == "water"]

# ───────────────────────── header + metric cards ─────────────────────────
st.title(f"🌿 {ctx['location']['city']}, {ctx['location']['country']}")
st.caption(f"📡 {ctx['data_source']}  ·  {ctx['location']['climate_name']} ({ctx['location']['climate_zone']})")

m = st.columns(6)
aqi = ctx["aqi"]["score"]
m[0].metric("AQI", f"{aqi}", ctx["aqi"]["category"],
            delta_color="inverse" if aqi > 100 else "normal")
m[1].metric("Temperature", f"{ctx['temperature_avg']}°C",
            f"feels {ctx['heat_index_avg']}°C")
m[2].metric("Humidity", f"{ctx['humidity_avg']}%")
m[3].metric("Rainfall", f"{water['annual_rainfall_mm']} mm/yr",
            water["stress_level"].split("(")[0].strip())
m[4].metric("Solar", f"{energy['solar_kwh_m2_day']} kWh/m²/d",
            energy["terrain"]["slope_class"].split("(")[0].strip())
m[5].metric("Density", f"{ctx['population_density']:,}/km²", ctx["density_class"])

tab_report, tab_map, tab_charts, tab_energy, tab_qa = st.tabs(
    ["📋 Report", "🗺 Satellite map", "📊 Impact charts", "⚡ Energy plan", "💬 Ask EcoGPT"])

# ───────────────────────── report ─────────────────────────
with tab_report:
    st.markdown(st.session_state["report"])
    st.download_button("⬇ Download report (Markdown)", st.session_state["report"],
                       file_name=f"ecogpt_{ctx['location']['city'].lower().replace(' ','_')}.md")

# ───────────────────────── map ─────────────────────────
with tab_map:
    if core.CAPS["folium"]:
        import folium
        from folium.plugins import MarkerCluster
        fmap = folium.Map(location=[lat, lon], zoom_start=14, tiles=None)
        folium.TileLayer("CartoDB positron", name="Street map").add_to(fmap)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="🛰 Satellite").add_to(fmap)
        top_species = [s["common"] for s in plant["species"][:4]]
        if open_plots:
            fg = folium.FeatureGroup(name=f"🌳 Plantable land ({len(open_plots)})")
            cl = MarkerCluster(name="📌 Planting sites")
            for s in open_plots:
                folium.Polygon(s["coords"], color="#2ecc40", weight=2, fill=True, fill_opacity=.35,
                    tooltip=f"🌳 {s['name']} — {s['area_m2']/1e4:.2f} ha ≈ {s['tree_capacity']:,} trees "
                            f"({', '.join(top_species)})").add_to(fg)
                cy = sum(c[0] for c in s["coords"])/len(s["coords"])
                cx = sum(c[1] for c in s["coords"])/len(s["coords"])
                folium.Marker([cy, cx], icon=folium.Icon(color="green", icon="leaf"),
                              tooltip=f"{s['name']}: ≈{s['tree_capacity']:,} trees").add_to(cl)
            fg.add_to(fmap); cl.add_to(fmap)
        if waterbodies:
            fgw = folium.FeatureGroup(name=f"💧 Water bodies ({len(waterbodies)})")
            for s in waterbodies:
                folium.Polygon(s["coords"], color="#0074d9", weight=2, fill=True, fill_opacity=.3,
                    tooltip=f"💧 {s['name']} — riparian buffer + reed beds").add_to(fgw)
            fgw.add_to(fmap)
        if rivers:
            fgr = folium.FeatureGroup(name=f"🌊 Rivers & streams ({len(rivers)})")
            for s in rivers:
                folium.PolyLine(s["coords"], color="#0d47a1", weight=4, opacity=.8,
                    tooltip=f"🌊 {s['name']} ({s.get('length_km',0)} km) — 10-30 m riparian buffer"
                    ).add_to(fgr)
            fgr.add_to(fmap)
        folium.Marker([lat, lon], tooltip=f"AQI {aqi} ({ctx['aqi']['category']})",
                      icon=folium.Icon(color="red" if aqi > 150 else "orange" if aqi > 100
                                       else "green")).add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        st.components.v1.html(fmap._repr_html_(), height=620)
        c1, c2, c3 = st.columns(3)
        c1.metric("Plantable plots", len(open_plots),
                  f"{sum(s['area_m2'] for s in open_plots)/1e4:.1f} ha")
        c2.metric("Tree capacity", f"{sum(s['tree_capacity'] for s in open_plots):,}")
        c3.metric("Rivers/streams", len(rivers),
                  f"{sum(s.get('length_km',0) for s in rivers):.1f} km mapped")
        if not spaces:
            st.warning("OSM land survey unavailable (offline?) — map shows base layers only.")
    else:
        st.warning("folium not installed — `pip install folium`")

# ───────────────────────── charts ─────────────────────────
with tab_charts:
    import matplotlib.pyplot as plt
    import math as _math
    GRN, RED, BLU, ORG, GRY = "#2e7d32", "#c62828", "#1565c0", "#ef6c00", "#9e9e9e"
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .25,
                         "axes.spines.top": False, "axes.spines.right": False})
    s_ = ctx["stats"]; gm = lambda c: s_.get(c, {}).get("mean", float("nan"))
    c1, c2 = st.columns(2)
    with c1:
        fig, a = plt.subplots(figsize=(6, 2.2)); left = 0
        for ceil_, col in [(50,"#00e400"),(100,"#ffff00"),(150,"#ff7e00"),
                           (200,"#ff0000"),(300,"#8f3f97"),(400,"#7e0023")]:
            a.barh(0, ceil_-left, left=left, color=col, height=.45); left = ceil_
        a.axvline(min(aqi, 400), color="black", lw=4)
        a.set(xlim=(0,400), yticks=[], title=f"AQI {aqi} — {ctx['aqi']['category']}")
        st.pyplot(fig, use_container_width=True)

        fig, a = plt.subplots(figsize=(6, 3))
        vals = [gm("RAWPM"), gm("DD"), gm("NO2")*1880, gm("CO")*1.145]
        who = [15, 45, 25, 4]
        ratio = [v/w if v == v else 0 for v, w in zip(vals, who)]
        a.bar(["PM2.5","PM10","NO2","CO"], ratio,
              color=[RED if r > 1 else GRN for r in ratio])
        a.axhline(1, color="black", ls="--"); a.set(title="Pollution vs WHO limits (× limit)")
        st.pyplot(fig, use_container_width=True)

        fig, a = plt.subplots(figsize=(6, 3))
        d_ = ctx["heat_index_avg"] - ctx["temperature_avg"]
        a.bar(["Air temp","Feels like"], [ctx["temperature_avg"], ctx["heat_index_avg"]],
              color=[GRY, RED if d_ > 3 else ORG])
        a.set(title=f"Heat island: +{d_:.1f}°C", ylabel="°C")
        st.pyplot(fig, use_container_width=True)
    with c2:
        fig, a = plt.subplots(figsize=(6, 3))
        yrs = [1,5,10,25]; vv = [carbon["cumulative_tonnes_co2e"][f"year_{y}"] for y in yrs]
        a.plot(yrs, vv, "o-", color=GRN, lw=2.5); a.fill_between(yrs, vv, alpha=.18, color=GRN)
        a.set(title=f"CO2 capture, {carbon['trees_modelled']:,} trees (t)", xlabel="year")
        st.pyplot(fig, use_container_width=True)

        fig, a = plt.subplots(figsize=(6, 3))
        base_n = {"Rural":50000,"Peri-urban":25000,"Urban":15000,"Dense Urban":10000}[ctx["density_class"]]
        scen = [(base_n,5),(base_n*3,10),(base_n*6,25)]
        sims = [core.simulate_intervention(aqi, n, n/plant["trees_per_hectare"], y) for n, y in scen]
        lbl = [f"{n//1000}k/{y}y" for n, y in scen]
        a.bar(lbl, [aqi]*3, color=GRY, label="today")
        a.bar(lbl, [x["projected_aqi"] for x in sims], color=GRN, label="projected")
        a.legend(); a.set(title="What-if: plantation scenarios → AQI")
        st.pyplot(fig, use_container_width=True)

        fig, a = plt.subplots(figsize=(6, 3))
        a.bar(["Solar kWh/m²/d","Wind m/s"], [energy["solar_kwh_m2_day"], energy["wind_ms"]],
              color=[ORG, BLU])
        a.axhline(3.5, color=ORG, ls="--", lw=1); a.axhline(4.0, color=BLU, ls=":", lw=1)
        a.set(title="Renewable resources vs viability thresholds")
        st.pyplot(fig, use_container_width=True)

# ───────────────────────── energy ─────────────────────────
with tab_energy:
    import pandas as pd
    st.subheader("Feasibility-checked interventions for this site")
    st.caption(f"Terrain: **{energy['terrain']['slope_class']}** · Solar **{energy['solar_kwh_m2_day']} "
               f"kWh/m²/day** · Wind **{energy['wind_ms']} m/s** · Grid **{energy['grid_emission_factor_kg_kwh']} kg CO2/kWh**")
    st.dataframe(pd.DataFrame(energy["interventions"]), use_container_width=True, hide_index=True)
    sp = energy["solar_plan"]
    st.subheader("☀️ Solar action plan")
    c = st.columns(3)
    c[0].metric("Rooftop potential", f"{sp['potential_mwp_per_km2']} MWp/km²")
    c[1].metric("Annual generation", f"{sp['annual_generation_gwh_per_km2']} GWh/yr/km²")
    c[2].metric("CO2 avoided", f"{sp['co2_avoided_t_per_km2_yr']:,} t/yr/km²")
    for step in sp["phased_rollout"]:
        st.markdown(f"- {step}")
    st.info(f"Siting rule: {sp['siting_rule']}")

# ───────────────────────── Q&A ─────────────────────────
with tab_qa:
    st.caption("RAG-grounded Q&A over the knowledge base + this location's context")
    if "chat" not in st.session_state: st.session_state["chat"] = []
    for role, msg in st.session_state["chat"]:
        st.chat_message(role).markdown(msg)
    if q := st.chat_input("e.g. which species absorb NO2? · how do khadins work?"):
        st.session_state["chat"].append(("user", q))
        st.chat_message("user").markdown(q)
        with st.spinner("retrieving…"):
            ans = core.ask_followup(q)
        st.session_state["chat"].append(("assistant", ans))
        st.chat_message("assistant").markdown(ans)
