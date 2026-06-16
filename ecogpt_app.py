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
    # ── LLM telemetry panel ───────────────────────────────────
    if core.LLM_LOG:
        with st.expander(f"🤖 LLM telemetry ({len(core.LLM_LOG)} calls)", expanded=False):
            _tot_in  = sum(e["tokens_in"]  for e in core.LLM_LOG)
            _tot_out = sum(e["tokens_out"] for e in core.LLM_LOG)
            _avg_lat = int(sum(e["latency_ms"] for e in core.LLM_LOG) / len(core.LLM_LOG))
            st.metric("Tokens in",  f"{_tot_in:,}")
            st.metric("Tokens out", f"{_tot_out:,}")
            st.metric("Avg latency", f"{_avg_lat} ms")
            import pandas as _pd_st
            st.dataframe(_pd_st.DataFrame(core.LLM_LOG)[
                ["ts", "backend", "model", "latency_ms",
                 "tokens_in", "tokens_out", "success"]
            ].rename(columns={"latency_ms": "ms", "tokens_in": "↑tok",
                               "tokens_out": "↓tok"}),
                use_container_width=True, hide_index=True)

if run and query.strip():
    loc = core.parse_location(query.strip())
    if loc is None:
        st.error("Couldn't identify that location — try a city name or `lat, lon` coordinates.")
    else:
        st.session_state["audience_used"] = audience
        with st.spinner(f"8 agents analysing {loc[2]} …"):
            report = core.ask_ecogpt(query.strip(), user_type=audience, polish=False,
                                     survey_radius_km=radius_km)
            st.session_state["report"] = report
            bundle = dict(core.ask_ecogpt.last)
            st.session_state["bundle"] = bundle
            st.session_state["loc"] = loc
            # spaces already fetched inside ask_ecogpt — reuse, no duplicate OSM call
            st.session_state["spaces"] = bundle.get("spaces", [])

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
_audience_badge = {"government": "🏛 Government/Policy", "ngo": "🤝 NGO/Community",
                   "researcher": "🔬 Researcher", "default": "🌍 General"
                   }.get(st.session_state.get("audience_used", "default"), "🌍 General")
st.title(f"🌿 {ctx['location']['city']}, {ctx['location']['country']}")

# Build caption: data source + live timestamp + LLM backend info
_live_ts = (ctx.get("live_env") or {}).get("data_ts", "")
_ts_part  = f"  ·  📅 {_live_ts[:16]}" if _live_ts else ""
_llm_part = ""
if core.LLM_LOG:
    _le = core.LLM_LOG[-1]
    _llm_part = (f"  ·  🤖 {core.LLM.kind}/{_le['model'].split('/')[-1]}"
                 f"  {_le['latency_ms']} ms  ↑{_le['tokens_in']} ↓{_le['tokens_out']} tok")
elif core.LLM.kind != "none":
    _llm_part = f"  ·  🤖 {core.LLM.kind} ({core.LLM.model})"
st.caption(f"📡 {ctx['data_source']}{_ts_part}"
           f"  ·  {ctx['location']['climate_name']} ({ctx['location']['climate_zone']})"
           f"  ·  {_audience_badge} report{_llm_part}")

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
    import numpy as np
    import math as _math

    # ── Modern EcoGPT palette ─────────────────────────────
    _C = dict(
        green      = "#2e7d32", green_lt = "#4caf50", green_pale = "#c8e6c9",
        red        = "#c62828", blue     = "#1565c0", blue_lt    = "#90caf9",
        orange     = "#e65100", orange_lt= "#ffb74d",
        grey       = "#bdbdbd", grey_dk  = "#616161",
        bg         = "#f4f8f5", panel    = "#ffffff",
        title      = "#1b5e20", text     = "#424242", text_lt    = "#9e9e9e",
    )
    plt.rcParams.update({
        "figure.facecolor"  : _C["bg"],   "axes.facecolor"    : _C["panel"],
        "axes.edgecolor"    : "#e0e0e0",  "axes.linewidth"    : 0.8,
        "axes.spines.top"   : False,      "axes.spines.right" : False,
        "axes.spines.left"  : False,      "axes.grid"         : True,
        "axes.grid.axis"    : "y",        "grid.color"        : "#efefef",
        "grid.linewidth"    : 0.9,        "axes.titlesize"    : 10,
        "axes.titleweight"  : "bold",     "axes.titlecolor"   : _C["title"],
        "axes.labelsize"    : 8.5,        "xtick.labelsize"   : 8,
        "ytick.labelsize"   : 8,          "xtick.color"       : _C["text_lt"],
        "ytick.color"       : _C["text_lt"], "legend.frameon"  : False,
        "font.family"       : "sans-serif",
    })

    def _lbl_st(a, bars, fmt="{:.1f}", fs=8.5):
        ymax = max((b.get_height() for b in bars), default=1) or 1
        for b in bars:
            h = b.get_height()
            if h != h or h == 0: continue
            a.text(b.get_x()+b.get_width()/2, h+ymax*0.035,
                   fmt.format(h), ha="center", va="bottom",
                   fontsize=fs, color=_C["text"], fontweight="bold")

    s_ = ctx["stats"]; gm = lambda c: s_.get(c, {}).get("mean", float("nan"))
    c1, c2 = st.columns(2)

    with c1:
        # ── AQI gauge ──────────────────────────────────
        fig, a = plt.subplots(figsize=(6, 2.4))
        fig.patch.set_facecolor(_C["bg"]); a.set_facecolor(_C["panel"])
        bands = [(50,"#57d98f"),(100,"#ffe566"),(150,"#ff9a3c"),
                 (200,"#ff4a4a"),(300,"#b05fcc"),(400,"#7e0023")]
        left = 0
        for ceil_, col in bands:
            a.barh(0, ceil_-left, left=left, color=col, height=0.44, edgecolor="none")
            left = ceil_
        nv = min(aqi, 400)
        a.axvline(nv, color="#212121", lw=3.5, zorder=5)
        a.scatter([nv], [0.28], color="#212121", s=70, zorder=6, clip_on=False)
        for cx, cl in [(25,"Good"),(75,"Moderate"),(125,"USG"),(175,"Unhealthy"),
                       (250,"V.Unhealthy"),(350,"Hazardous")]:
            a.text(cx, -0.28, cl, ha="center", va="top", fontsize=5.5, color=_C["grey_dk"])
        a.text(nv, 0.5, f"  {aqi}  \n{ctx['aqi']['category']}",
               ha="center", va="bottom", fontsize=9, fontweight="bold", color=_C["title"],
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         edgecolor=_C["green_pale"], linewidth=1))
        a.set(xlim=(0,400), ylim=(-0.5,0.85), yticks=[])
        a.spines["bottom"].set_visible(False); a.spines["left"].set_visible(False)
        a.grid(False); a.set_title("Air Quality Index", pad=4)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── Pollution vs WHO ────────────────────────────
        fig, a = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor(_C["bg"]); a.set_facecolor(_C["panel"])
        vals = [gm("RAWPM"), gm("DD"), gm("NO2")*1880, gm("CO")*1.145]
        who  = [15, 45, 25, 4]
        ratio = [v/w if (v==v and w) else 0 for v, w in zip(vals, who)]
        b2 = a.bar(["PM₂.₅","PM₁₀","NO₂","CO"], ratio,
                   color=[_C["red"] if r>1 else _C["green_lt"] for r in ratio],
                   edgecolor="none", width=0.6, zorder=3)
        a.axhspan(0, 1, alpha=0.07, color=_C["green"], zorder=0)
        a.axhline(1, color=_C["grey_dk"], ls="--", lw=1.5, zorder=4)
        a.text(3.46, 1.06, "WHO limit", fontsize=7.5, color=_C["grey_dk"])
        rmax = max(ratio or [0.1]) or 0.1
        for b, v, r in zip(b2, vals, ratio):
            if v==v and v:
                a.text(b.get_x()+b.get_width()/2, r+rmax*0.05,
                       f"{v:.0f}\n({r:.1f}×)", ha="center", va="bottom",
                       fontsize=7.5, color=_C["red"] if r>1 else _C["green"],
                       fontweight="bold")
        a.set(title="Pollutants vs WHO 24-h guideline", ylabel="× WHO limit")
        a.yaxis.grid(True, color="#efefef", zorder=0)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── Heat island ─────────────────────────────────
        fig, a = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor(_C["bg"]); a.set_facecolor(_C["panel"])
        d_ = ctx["heat_index_avg"] - ctx["temperature_avg"]
        b5 = a.bar(["Air temperature", "Feels like\n(heat index)"],
                   [ctx["temperature_avg"], ctx["heat_index_avg"]],
                   color=[_C["grey"], _C["red"] if d_>3 else _C["orange"]],
                   edgecolor="none", width=0.45, zorder=3)
        _lbl_st(a, b5, "{:.1f}°C", fs=9.5)
        hi = ctx["heat_index_avg"]
        a.annotate(f"+{d_:.1f}°C",
                   xy=(1, hi), xytext=(0.35, hi+max(hi*0.07, 1.5)),
                   arrowprops=dict(arrowstyle="->", color=_C["red"], lw=1.3),
                   fontsize=9, color=_C["red"], fontweight="bold", ha="center")
        a.set(title=f"Heat island — +{d_:.1f}°C feels-like", ylabel="°C")
        a.yaxis.grid(True, color="#efefef", zorder=0)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with c2:
        # ── CO₂ projection ──────────────────────────────
        fig, a = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor(_C["bg"]); a.set_facecolor(_C["panel"])
        yrs = [1,5,10,25]
        vv  = [carbon["cumulative_tonnes_co2e"][f"year_{y}"] for y in yrs]
        a.fill_between(yrs, vv, color=_C["green"], alpha=0.12, zorder=1)
        a.plot(yrs, vv, "-", color=_C["green"], lw=2.5, zorder=3)
        a.scatter(yrs, vv, color="white", s=55, zorder=4,
                  edgecolors=_C["green"], linewidths=2)
        for x, y in zip(yrs, vv):
            a.annotate(f"{y:,.0f} t", (x, y),
                       textcoords="offset points", xytext=(0,9),
                       ha="center", fontsize=8, color=_C["green"], fontweight="bold")
        a.set(title=f"CO₂ captured — {carbon['trees_modelled']:,} trees",
              xlabel="Years after planting", ylabel="Cumulative t CO₂e")
        a.set_xticks(yrs); a.yaxis.grid(True, color="#efefef", zorder=0)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── What-if AQI ─────────────────────────────────
        fig, a = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor(_C["bg"]); a.set_facecolor(_C["panel"])
        base_n = carbon["trees_modelled"]
        survey_area_ha = math.pi * radius_km ** 2 * 100
        scen = [(base_n,5),(base_n*3,10),(base_n*6,25)]
        sims = [core.simulate_intervention(aqi, n, n/plant["trees_per_hectare"], y,
                                           urban_area_ha=survey_area_ha)
                for n, y in scen]
        lbl6 = [(f"{n//1000}k trees\n{y} yrs" if n>=1000
                 else f"{n} trees\n{y} yrs") for n, y in scen]
        x6 = np.arange(len(scen)); w6 = 0.34
        a.bar(x6-w6/2, [aqi]*len(scen), w6,
              color=_C["grey"], alpha=0.65, edgecolor="none", label="Today")
        a.bar(x6+w6/2, [s["projected_aqi"] for s in sims], w6,
              color=_C["green_lt"], edgecolor="none", label="Projected")
        for i, s_ in enumerate(sims):
            a.text(x6[i]+w6/2, s_["projected_aqi"]+1.5,
                   f"−{s_['aqi_reduction']}", ha="center", va="bottom",
                   fontsize=8.5, fontweight="bold", color=_C["green"])
        a.set_xticks(list(x6)); a.set_xticklabels(lbl6, fontsize=8)
        a.legend(loc="upper right"); a.set(title="What-if: plantation → AQI impact")
        a.yaxis.grid(True, color="#efefef", zorder=0)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── Renewables ───────────────────────────────────
        fig, a = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor(_C["bg"]); a.set_facecolor(_C["panel"])
        b_rv = a.bar(["Solar\n(kWh/m²/d)", "Wind\n(m/s)"],
                     [energy["solar_kwh_m2_day"], energy["wind_ms"]],
                     color=[_C["orange"], _C["blue"]],
                     edgecolor="none", width=0.45, zorder=3)
        a.axhline(3.5, xmin=0.05, xmax=0.47, color=_C["orange"], ls="--", lw=1.4)
        a.axhline(4.0, xmin=0.53, xmax=0.95, color=_C["blue"],   ls="--", lw=1.4)
        a.text(-0.33, 3.62, "≥3.5 viable", fontsize=8, color=_C["orange"])
        a.text( 0.67, 4.12, "≥4.0 viable", fontsize=8, color=_C["blue"])
        _lbl_st(a, b_rv, "{:.2f}", fs=10)
        a.set(title="Renewable resources vs viability thresholds")
        a.yaxis.grid(True, color="#efefef", zorder=0)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

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
    st.caption(f"RAG-grounded Q&A · knowledge base + {ctx['location']['city']} pipeline context")
    st.info("💡 Ask anything about this location: *which species fight NO2?* · "
            "*how do khadins work?* · *how much water can a rooftop harvest?* · "
            "*is solar viable here?* · *what soil improvements are needed?*")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for role, msg in st.session_state["chat"]:
        st.chat_message(role).markdown(msg)
    if q := st.chat_input(f"Ask about {ctx['location']['city']}…"):
        st.session_state["chat"].append(("user", q))
        st.chat_message("user").markdown(q)
        with st.spinner("Analysing…"):
            try:
                ans = core.ask_followup(q)
            except Exception as e:
                ans = f"⚠️ Error: {e}. Try rephrasing or run the analysis again."
        st.session_state["chat"].append(("assistant", ans))
        st.chat_message("assistant").markdown(ans)
    if st.session_state["chat"]:
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state["chat"] = []
            st.rerun()
