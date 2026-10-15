from pathlib import Path
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.data_utils import load_or_create_data, SENSOR_COLUMNS, sensor_status, generate_sensor_data
from src.model import train_model, load_model, predict_rows, FEATURES

st.set_page_config(page_title="Sentinel | Predictive Maintenance", page_icon="🛠️", layout="wide", initial_sidebar_state="expanded")

# ---------- Visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1220; color: #e8eef8; }
[data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #233149; }
[data-testid="stSidebar"] * { color: #dbe7f7; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; color: #f5f8ff; }
p, label, .stCaption { color: #9fb0c9; }
div[data-testid="stMetric"] { background: linear-gradient(135deg,#14243a,#111c2e); border: 1px solid #263952; padding: 18px 18px; border-radius: 14px; }
div[data-testid="stMetricLabel"] { color: #9fb0c9; }
div[data-testid="stMetricValue"] { color: #f6f9ff; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; }
.panel { background: #111e31; border: 1px solid #253750; border-radius: 16px; padding: 18px 20px; }
.eyebrow { color:#65d7c3; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; }
.hero { background: radial-gradient(circle at 90% 0%, #1b5360 0, transparent 34%), linear-gradient(115deg,#14273d,#111b2c); border:1px solid #29445a; padding:26px 28px; border-radius:20px; margin-bottom:22px; }
.hero h1 { margin:4px 0 8px 0; font-size:34px; }
.hero p { max-width:780px; font-size:14px; }
div.stButton > button { border-radius: 10px; border:1px solid #2d536b; background:#1c9b8b; color:white; font-weight:700; }
div.stButton > button:hover { background:#27b6a4; border-color:#27b6a4; color:white; }
div[data-testid="stDataFrame"] { border:1px solid #263952; border-radius:12px; overflow:hidden; }
hr { border-color:#24344b; }
.small-muted { color:#8da2bd; font-size:12px; }
</style>
""", unsafe_allow_html=True)

def chart_theme(fig, height=330):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b9c8dd", family="DM Sans"),
        margin=dict(l=10,r=10,t=35,b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#233149", zerolinecolor="#233149"),
        yaxis=dict(gridcolor="#233149", zerolinecolor="#233149")
    )
    return fig

def status_badge(status):
    colors = {"Healthy":"#1db98c", "Warning":"#e6a23c", "Critical":"#f05d6c"}
    return f'<span style="background:{colors.get(status,"#8795a8")}22;color:{colors.get(status,"#8795a8")};padding:5px 10px;border-radius:99px;font-weight:700;font-size:12px;border:1px solid {colors.get(status,"#8795a8")}66">{status}</span>'

@st.cache_data(show_spinner=False)
def get_data():
    return load_or_create_data()

def app_header(title, subtitle, section="OPERATIONS CENTER"):
    st.markdown(f'<div class="eyebrow">{section}</div><h1 style="margin-top:3px">{title}</h1><p style="margin-top:-8px">{subtitle}</p>', unsafe_allow_html=True)

def run_app():
    df = get_data()
    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = load_model()
    if "sim_rows" not in st.session_state:
        st.session_state.sim_rows = []
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.5

    with st.sidebar:
        st.markdown("## 🛠️ SENTINEL")
        st.caption("AI · IoT · Asset Intelligence")
        st.divider()
        page = st.radio("WORKSPACE", ["Overview", "Live Monitor", "Failure Predictions", "Analytics", "Dataset Explorer", "Model Center"], label_visibility="visible")
        st.divider()
        st.markdown("**SYSTEM STATUS**")
        st.markdown("🟢 Demo telemetry online")
        st.markdown("🟢 Local ML pipeline ready" if st.session_state.model_bundle else "🟠 Model needs training")
        st.caption("Demo environment • synthetic sensor data")
        st.divider()
        st.markdown('<div class="small-muted">SENTINEL PM v2.0<br/>Built for learning and portfolio demos</div>', unsafe_allow_html=True)

    # Data enriched with predictions whenever a trained model exists
    bundle = st.session_state.model_bundle
    if bundle:
        work = df.copy()
        for col in FEATURES:
            if col not in work:
                work[col] = 0
        try:
            preds, probs = predict_rows(bundle, work[FEATURES].fillna(work[FEATURES].median()))
            work["risk_probability"] = probs
            work["predicted_failure"] = preds
        except Exception:
            work["risk_probability"] = 0.0
            work["predicted_failure"] = 0
    else:
        work = df.copy()
        # Transparent heuristic fallback until the ML model is trained.
        risk = (
            (work["temperature"].fillna(50) > 72).astype(float)*.30 +
            (work["vibration"].fillna(1.5) > 3.5).astype(float)*.32 +
            (work["current"].fillna(6) > 10.5).astype(float)*.20 +
            (work["pressure"].fillna(3) < 2.5).astype(float)*.18
        )
        work["risk_probability"] = risk.clip(0,1)
        work["predicted_failure"] = (work["risk_probability"] >= st.session_state.threshold).astype(int)

    latest = work.sort_values("timestamp").groupby("machine_id", as_index=False).tail(1).copy()
    latest["status"] = latest.apply(sensor_status, axis=1)
    failures = int(latest["predicted_failure"].sum())
    critical = int((latest["status"] == "Critical").sum())
    warning = int((latest["status"] == "Warning").sum())
    healthy = int((latest["status"] == "Healthy").sum())

    if page == "Overview":
        st.markdown('<div class="hero"><div class="eyebrow">PREDICTIVE MAINTENANCE PLATFORM</div><h1>See failure coming.<br/>Keep operations moving.</h1><p>Monitor simulated industrial assets, surface abnormal sensor patterns, and explore machine-learning risk scores in one operational workspace.</p></div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Assets monitored", f"{latest.machine_id.nunique():,}", "Telemetry connected")
        c2.metric("Healthy assets", healthy, f"{healthy/max(len(latest),1):.0%} of fleet")
        c3.metric("Needs attention", warning + critical, f"{critical} critical")
        c4.metric("AI risk flags", failures, "Current latest readings")
        left,right = st.columns([1.55,1])
        with left:
            st.markdown("### Sensor telemetry")
            metric = st.selectbox("Signal", ["temperature","vibration","current","pressure"], key="overview_signal")
            chart = work.sort_values("timestamp").tail(300)
            fig = px.line(chart, x="timestamp", y=metric, color="machine_id", title=f"{metric.title()} · recent telemetry")
            st.plotly_chart(chart_theme(fig,350), use_container_width=True)
        with right:
            st.markdown("### Fleet health")
            health = latest["status"].value_counts().reindex(["Healthy","Warning","Critical"], fill_value=0).reset_index()
            health.columns = ["Status","Assets"]
            fig = px.pie(health, names="Status", values="Assets", hole=.68, color="Status",
                         color_discrete_map={"Healthy":"#1db98c","Warning":"#e6a23c","Critical":"#f05d6c"})
            fig.update_traces(textposition="outside", textinfo="label+value")
            st.plotly_chart(chart_theme(fig,300), use_container_width=True)
            st.markdown('<div class="panel"><div class="eyebrow">OPERATIONS NOTE</div><p style="margin-bottom:0;color:#c5d4e8">Risk flags are decision-support signals, not a guarantee of failure. Review sensor context before maintenance action.</p></div>', unsafe_allow_html=True)
        st.markdown("### Asset watchlist")
        display = latest[["machine_id","timestamp","temperature","vibration","current","pressure","risk_probability","status"]].copy()
        display["risk_probability"] = (display["risk_probability"]*100).round(1).astype(str)+"%"
        display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%d %b %H:%M")
        display = display.rename(columns={"machine_id":"Asset","timestamp":"Last update","temperature":"Temp °C","vibration":"Vibration","current":"Current A","pressure":"Pressure bar","risk_probability":"AI risk","status":"Status"})
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("Dataset is synthetic and generated locally. Results demonstrate workflow, not validated industrial performance.")

    elif page == "Live Monitor":
        app_header("Live monitor", "Generate fresh virtual sensor readings and inspect risk in near real time.")
        st.markdown('<div class="panel"><div class="eyebrow">VIRTUAL TELEMETRY SIMULATOR</div><p>Each simulation tick creates new readings for a selected asset. No hardware is required; the values are synthetic.</p></div>', unsafe_allow_html=True)
        a,b,c = st.columns([1,1,1.4])
        with a: asset = st.selectbox("Asset", ["MTR-001","PMP-002","FAN-003","CMP-004"])
        with b: mode = st.selectbox("Operating condition", ["Normal operation","Elevated load","Degradation event"])
        with c: st.session_state.threshold = st.slider("Failure risk threshold", 0.2, 0.9, float(st.session_state.threshold), 0.05)
        if st.button("＋ Generate sensor reading", use_container_width=True):
            rng = np.random.default_rng()
            if mode == "Normal operation":
                vals = [rng.normal(48,4), rng.normal(1.5,.25), rng.normal(6.5,.7), rng.normal(3.5,.2), rng.normal(.55,.08), rng.integers(30,700)]
            elif mode == "Elevated load":
                vals = [rng.normal(68,4), rng.normal(3.0,.4), rng.normal(10.0,.8), rng.normal(2.8,.2), rng.normal(.9,.05), rng.integers(200,1300)]
            else:
                vals = [rng.normal(82,3), rng.normal(4.4,.35), rng.normal(12.4,.8), rng.normal(2.25,.15), rng.normal(1.05,.05), rng.integers(700,1800)]
            row = {"timestamp":pd.Timestamp.now(), "machine_id":asset, "temperature":round(float(vals[0]),2),
                   "vibration":round(float(vals[1]),3), "current":round(float(vals[2]),3), "pressure":round(float(vals[3]),3),
                   "load":round(float(np.clip(vals[4],.1,1.2)),3), "age_days":int(vals[5])}
            if bundle:
                _, p = predict_rows(bundle, pd.DataFrame([row])[FEATURES])
                row["risk_probability"] = float(p[0])
            else:
                row["risk_probability"] = float(np.clip((row["temperature"]>72)*.3+(row["vibration"]>3.5)*.32+(row["current"]>10.5)*.2+(row["pressure"]<2.5)*.18,0,1))
            row["predicted_failure"] = int(row["risk_probability"] >= st.session_state.threshold)
            row["status"] = sensor_status(row)
            st.session_state.sim_rows.insert(0,row)
            st.session_state.sim_rows = st.session_state.sim_rows[:100]
        if st.session_state.sim_rows:
            r = st.session_state.sim_rows[0]
            st.markdown("### Latest simulated reading")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Temperature", f'{r["temperature"]:.1f} °C')
            m2.metric("Vibration", f'{r["vibration"]:.2f} mm/s')
            m3.metric("Current", f'{r["current"]:.2f} A')
            m4.metric("Pressure", f'{r["pressure"]:.2f} bar')
            st.markdown(status_badge(r["status"]), unsafe_allow_html=True)
            if r["predicted_failure"]:
                st.error(f'AI risk alert: estimated failure probability {r["risk_probability"]:.0%}. Inspect asset and validate readings.')
            else:
                st.success(f'No failure flag at current threshold. Estimated risk: {r["risk_probability"]:.0%}.')
            simdf = pd.DataFrame(st.session_state.sim_rows)
            st.markdown("### Simulation history")
            fig = px.line(simdf.iloc[::-1], x="timestamp", y="risk_probability", color="machine_id", markers=True, title="Estimated risk over generated readings")
            fig.add_hline(y=st.session_state.threshold, line_dash="dash", line_color="#f05d6c", annotation_text="Alert threshold")
            st.plotly_chart(chart_theme(fig,330), use_container_width=True)
            st.dataframe(simdf[["timestamp","machine_id","temperature","vibration","current","pressure","risk_probability","status"]], use_container_width=True, hide_index=True)
            csv = simdf.to_csv(index=False).encode("utf-8")
            st.download_button("Download simulation CSV", csv, "virtual_sensor_readings.csv", "text/csv")
        else:
            st.info("Choose an operating condition and click **Generate sensor reading** to start the virtual simulation.")

    elif page == "Failure Predictions":
        app_header("Failure predictions", "Prioritize assets using estimated model probability and sensor context.")
        threshold = st.slider("Alert threshold", .1, .9, float(st.session_state.threshold), .05, key="pred_threshold")
        st.session_state.threshold = threshold
        pred = work.copy()
        pred["alert"] = np.where(pred["risk_probability"] >= threshold, "Flagged", "Monitor")
        col1,col2 = st.columns([1.4,1])
        with col1:
            fig = px.scatter(pred.tail(700), x="temperature", y="vibration", color="alert", size="risk_probability",
                             hover_data=["machine_id","current","pressure","risk_probability"],
                             color_discrete_map={"Flagged":"#f05d6c","Monitor":"#1db98c"}, title="Sensor risk map")
            st.plotly_chart(chart_theme(fig,380), use_container_width=True)
        with col2:
            counts = pred["alert"].value_counts().rename_axis("Decision").reset_index(name="Readings")
            fig = px.bar(counts, x="Decision", y="Readings", color="Decision",
                         color_discrete_map={"Flagged":"#f05d6c","Monitor":"#1db98c"}, title="Readings by decision")
            st.plotly_chart(chart_theme(fig,380), use_container_width=True)
        flagged = pred[pred["risk_probability"] >= threshold].sort_values("risk_probability", ascending=False)
        st.markdown("### Highest-risk readings")
        if flagged.empty:
            st.success("No readings exceed the selected threshold.")
        else:
            out = flagged[["timestamp","machine_id","temperature","vibration","current","pressure","risk_probability","failure"]].head(100).copy()
            out["risk_probability"] = (out["risk_probability"]*100).round(1).astype(str)+"%"
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("Export flagged readings", flagged.to_csv(index=False).encode(), "flagged_readings.csv", "text/csv")
        st.caption("Probability estimates depend on the demo dataset and model. Do not use for safety-critical decisions.")

    elif page == "Analytics":
        app_header("Fleet analytics", "Explore operating patterns, failure labels, and model risk trends.")
        f1,f2 = st.columns(2)
        with f1:
            daily = work.copy()
            daily["day"] = pd.to_datetime(daily["timestamp"]).dt.date
            trend = daily.groupby("day", as_index=False).agg(mean_temp=("temperature","mean"), mean_vibration=("vibration","mean"), labelled_failures=("failure","sum"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["day"], y=trend["mean_temp"], name="Mean temperature", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=trend["day"], y=trend["mean_vibration"], name="Mean vibration", yaxis="y2", line=dict(width=2,dash="dot")))
            fig.update_layout(title="Sensor trends by day", yaxis=dict(title="Temperature °C"), yaxis2=dict(title="Vibration",overlaying="y",side="right",showgrid=False))
            st.plotly_chart(chart_theme(fig,350), use_container_width=True)
        with f2:
            by_machine = work.groupby("machine_id", as_index=False).agg(avg_temp=("temperature","mean"), avg_vibration=("vibration","mean"), failure_rate=("failure","mean"), avg_risk=("risk_probability","mean"))
            fig = px.bar(by_machine, x="machine_id", y="failure_rate", color="avg_risk", title="Labelled failure rate by asset",
                         color_continuous_scale=["#1db98c","#e6a23c","#f05d6c"])
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(chart_theme(fig,350), use_container_width=True)
        f3,f4 = st.columns(2)
        with f3:
            corr = work[FEATURES+["failure"]].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Tealrose", title="Sensor / target correlation")
            st.plotly_chart(chart_theme(fig,390), use_container_width=True)
        with f4:
            fig = px.histogram(work, x="temperature", color="failure", nbins=32, barmode="overlay",
                               color_discrete_map={0:"#4b9bd8",1:"#f05d6c"}, title="Temperature distribution by synthetic label")
            st.plotly_chart(chart_theme(fig,390), use_container_width=True)
        st.info("Analytics are computed from the locally generated synthetic dataset. They are useful for demonstrating the pipeline, not establishing real-world failure rates.")

    elif page == "Dataset Explorer":
        app_header("Dataset explorer", "Inspect, filter, and export the sensor telemetry used by this demo.")
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Missing sensor values", f"{int(df[SENSOR_COLUMNS].isna().sum().sum()):,}")
        c3.metric("Synthetic failure labels", f'{int(df["failure"].sum()):,}')
        machine = st.multiselect("Filter assets", sorted(df["machine_id"].unique()), default=sorted(df["machine_id"].unique()))
        label = st.selectbox("Label filter", ["All readings","Failure label = 1","Failure label = 0"])
        filtered = df[df["machine_id"].isin(machine)].copy()
        if label == "Failure label = 1": filtered = filtered[filtered["failure"] == 1]
        if label == "Failure label = 0": filtered = filtered[filtered["failure"] == 0]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)
        st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "iot_sensor_data_filtered.csv", "text/csv")
        with st.expander("Data dictionary"):
            st.markdown("""
            - `timestamp`: synthetic five-minute sample timestamp
            - `machine_id`: simulated asset identifier
            - `age_days`: simulated asset age
            - `load`: simulated normalized operating load
            - `temperature`: degrees Celsius
            - `vibration`: simulated vibration magnitude (arbitrary demo scale)
            - `current`: simulated current in amperes
            - `pressure`: simulated pressure in bar
            - `failure`: synthetic binary label generated from sensor-risk rules
            """)

    elif page == "Model Center":
        app_header("Model center", "Train, evaluate, and inspect the local Random Forest classifier.")
        st.markdown('<div class="panel"><div class="eyebrow">MODEL CARD</div><p><b>Algorithm</b> · Random Forest classifier<br/><b>Features</b> · Temperature, vibration, current, pressure, load, asset age<br/><b>Preprocessing</b> · Median imputation in a scikit-learn pipeline<br/><b>Validation</b> · Stratified holdout split<br/><b>Target</b> · Synthetic failure label</p></div>', unsafe_allow_html=True)
        if st.button("Train / retrain model", use_container_width=True):
            with st.spinner("Training and evaluating the Random Forest model..."):
                pipeline, metrics = train_model(df)
                st.session_state.model_bundle = load_model()
                st.success("Model trained and saved to models/predictive_maintenance_model.joblib")
                st.rerun()
        bundle = st.session_state.model_bundle
        if bundle:
            metrics = bundle["metrics"]
            st.markdown("### Evaluation metrics")
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Accuracy", f'{metrics["accuracy"]:.1%}')
            m2.metric("Precision", f'{metrics["precision"]:.1%}')
            m3.metric("Recall", f'{metrics["recall"]:.1%}')
            m4.metric("F1 score", f'{metrics["f1"]:.3f}')
            m5.metric("ROC AUC", f'{metrics["roc_auc"]:.3f}')
            cm = np.array(metrics["confusion_matrix"])
            fig = go.Figure(data=go.Heatmap(z=cm, x=["Predicted healthy","Predicted failure"], y=["Actual healthy","Actual failure"], text=cm, texttemplate="%{text}", colorscale="Teal"))
            fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(chart_theme(fig,330), use_container_width=True)
            model = bundle["pipeline"].named_steps["model"]
            imp = pd.DataFrame({"Feature":FEATURES,"Importance":model.feature_importances_}).sort_values("Importance",ascending=True)
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Feature importance", color="Importance", color_continuous_scale="Teal")
            st.plotly_chart(chart_theme(fig,340), use_container_width=True)
            st.caption(f'Train rows: {metrics["train_size"]:,} · Test rows: {metrics["test_size"]:,}')
        else:
            st.warning("No saved model detected yet. Train the model above; the app will use a transparent threshold-based fallback until then.")
        st.markdown("### Test a custom reading")
        with st.form("custom_prediction"):
            x1,x2,x3 = st.columns(3)
            temp = x1.number_input("Temperature °C", 0.0, 150.0, 74.0)
            vib = x2.number_input("Vibration", 0.0, 20.0, 3.8)
            current = x3.number_input("Current A", 0.0, 100.0, 11.0)
            x4,x5,x6 = st.columns(3)
            pressure = x4.number_input("Pressure bar", 0.0, 20.0, 2.6)
            load = x5.number_input("Load (0–1.2)", 0.0, 1.2, 0.85)
            age = x6.number_input("Asset age (days)", 0, 5000, 900)
            submitted = st.form_submit_button("Run prediction")
        if submitted:
            sample = pd.DataFrame([{"temperature":temp,"vibration":vib,"current":current,"pressure":pressure,"load":load,"age_days":age}])
            if bundle:
                pred, prob = predict_rows(bundle, sample)
                risk = float(prob[0]); is_failure = int(pred[0])
            else:
                risk = float(np.clip((temp>72)*.3+(vib>3.5)*.32+(current>10.5)*.2+(pressure<2.5)*.18,0,1))
                is_failure = int(risk >= st.session_state.threshold)
            if is_failure: st.error(f"Failure risk flag · estimated risk {risk:.1%}")
            else: st.success(f"No failure flag · estimated risk {risk:.1%}")
            st.progress(min(max(risk,0),1), text="Estimated failure probability")

    st.markdown("---")
    st.markdown('<div class="small-muted">SENTINEL PM · Portfolio demo · Synthetic data only · Not certified for operational decisions</div>', unsafe_allow_html=True)
