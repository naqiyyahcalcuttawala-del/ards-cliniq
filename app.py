import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARDS ClinIQ — ICU Decision Support",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background-color: #0E1117; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
  }
  section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem; color: #8b949e;
  }

  /* KPI cards */
  .kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color .2s;
  }
  .kpi-card:hover { border-color: #3b82f6; }
  .kpi-title { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .4rem; }
  .kpi-value { font-size: 2rem; font-weight: 700; line-height: 1; }
  .kpi-delta { font-size: 0.78rem; margin-top: .4rem; }
  .kpi-blue  { color: #3b82f6; }
  .kpi-red   { color: #ef4444; }
  .kpi-yellow{ color: #f59e0b; }
  .kpi-green { color: #10b981; }

  /* Section cards */
  .section-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
  }
  .section-header {
    font-size: 1rem; font-weight: 600; color: #e6edf3;
    margin-bottom: 1rem; padding-bottom: .6rem;
    border-bottom: 1px solid #21262d;
  }

  /* Prediction pill */
  .pred-high   { background:#ef444420; color:#ef4444; border:1px solid #ef4444; border-radius:8px; padding:.3rem .8rem; font-weight:600; }
  .pred-medium { background:#f59e0b20; color:#f59e0b; border:1px solid #f59e0b; border-radius:8px; padding:.3rem .8rem; font-weight:600; }
  .pred-low    { background:#10b98120; color:#10b981; border:1px solid #10b981; border-radius:8px; padding:.3rem .8rem; font-weight:600; }

  /* SBAR card */
  .sbar-section { background:#1c2128; border-left:3px solid #3b82f6; border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin-bottom:.8rem; }
  .sbar-label   { font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#3b82f6; margin-bottom:.4rem; }
  .sbar-text    { font-size:.88rem; color:#c9d1d9; line-height:1.6; }

  /* Metric override */
  [data-testid="stMetricValue"]  { color: #e6edf3 !important; font-size: 1.8rem !important; }
  [data-testid="stMetricLabel"]  { color: #8b949e !important; }

  /* HR */
  hr { border-color: #21262d; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }
  .stTabs [data-baseweb="tab"]      { color: #8b949e; }
  .stTabs [aria-selected="true"]    { color: #3b82f6 !important; border-bottom-color: #3b82f6 !important; }

  /* Buttons */
  .stButton > button {
    background: #3b82f6; color: white; border: none; border-radius: 8px;
    font-weight: 600; font-size: .88rem; padding: .5rem 1.5rem;
  }
  .stButton > button:hover { background: #2563eb; }

  /* Expander */
  details { border: 1px solid #21262d !important; border-radius: 8px !important; background: #161b22 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Data & Model loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "ARDS_ICU_V2_15000_final_buan305.csv"))
    # Impute missing values
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object"]).columns
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
    return df

@st.cache_resource
def train_models(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error
    from sklearn.cluster import KMeans

    # Features
    feat_cols = [
        'age','bmi','heart_rate_d0','map_d0','respiratory_rate_d0','spo2_d0',
        'heart_rate_d3','map_d3','respiratory_rate_d3','spo2_d3',
        'pao2_fio2_ratio_d0','fio2_d0','peep_d0','mean_airway_pressure_d0',
        'pao2_fio2_ratio_d3','fio2_d3','peep_d3','mean_airway_pressure_d3',
        'lactate_d0','crp_d0','albumin_d0','platelet_d0','bicarbonate_d0',
        'creatinine_d0','bilirubin_d0','wbc_d0',
        'lactate_d3','crp_d3','albumin_d3','platelet_d3','bicarbonate_d3',
        'creatinine_d3','bilirubin_d3','wbc_d3',
        'sofa_score_d0','sofa_score_d3','delta_sofa','delta_lactate',
        'delta_pf_ratio','delta_creatinine','delta_crp','shock_index',
        'organ_failure_count','comorbidity_count','mechanical_ventilation_days',
        'vasopressor_duration',
        'hypertension','diabetes','copd','ckd','cardiovascular_disease','liver_disease',
        'vasopressor_use_d0','vasopressor_use_d3','high_risk_comorbidity_flag',
    ]
    # Encode sex, smoking, ventilation
    df2 = df.copy()
    for c in ['sex','smoking_status','ventilation_type']:
        le = LabelEncoder()
        df2[c+"_enc"] = le.fit_transform(df2[c])
    feat_cols += ['sex_enc','smoking_status_enc','ventilation_type_enc']

    X = df2[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Mortality model ──────────────────────────────────────────────────────
    y_mort = df2['mortality_60d'].values
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y_mort, test_size=.2, random_state=42, stratify=y_mort)
    mort_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    mort_model.fit(Xtr, ytr)
    mort_probs = mort_model.predict_proba(Xte)[:,1]
    mort_preds = mort_model.predict(Xte)
    mort_auc = roc_auc_score(yte, mort_probs)
    mort_cm = confusion_matrix(yte, mort_preds)

    # ── LOS model ────────────────────────────────────────────────────────────
    y_los = df2['icu_los_days'].values
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X_scaled, y_los, test_size=.2, random_state=42)
    los_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    los_model.fit(Xtr2, ytr2)
    los_preds = los_model.predict(Xte2)
    los_rmse = np.sqrt(mean_squared_error(yte2, los_preds))
    los_mae = mean_absolute_error(yte2, los_preds)

    # ── Risk engine model ────────────────────────────────────────────────────
    le_risk = LabelEncoder()
    y_risk = le_risk.fit_transform(df2['risk_category'].values)
    Xtr3, Xte3, ytr3, yte3 = train_test_split(X_scaled, y_risk, test_size=.2, random_state=42, stratify=y_risk)
    risk_model = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    risk_model.fit(Xtr3, ytr3)
    risk_preds = risk_model.predict(Xte3)
    risk_auc = roc_auc_score(yte3, risk_model.predict_proba(Xte3), multi_class='ovr')

    # ── Clustering ───────────────────────────────────────────────────────────
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    return {
        "feat_cols": feat_cols,
        "scaler": scaler,
        "mort_model": mort_model,
        "los_model": los_model,
        "risk_model": risk_model,
        "le_risk": le_risk,
        "km": km,
        "mort_auc": mort_auc,
        "mort_cm": mort_cm,
        "los_rmse": los_rmse,
        "los_mae": los_mae,
        "risk_auc": risk_auc,
        "Xte_mort": Xte,
        "yte_mort": yte,
        "mort_probs": mort_probs,
        "mort_preds": mort_preds,
        "Xte_los": Xte2,
        "yte_los": yte2,
        "los_pred_vals": los_preds,
        "X_scaled": X_scaled,
        "cluster_labels": cluster_labels,
    }

# ─── Plotly theme helpers ─────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", family="Inter"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    margin=dict(l=40, r=20, t=40, b=40),
)

def dark_fig(fig):
    fig.update_layout(**DARK_LAYOUT)
    return fig

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 1.4rem;'>
      <div style='font-size:1.5rem;font-weight:800;color:#3b82f6;letter-spacing:-.02em;'>🫁 ARDS ClinIQ</div>
      <div style='font-size:.75rem;color:#8b949e;margin-top:.2rem;'>ICU Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Overview",
        "🧠  Mortality Intelligence",
        "⏱  LOS Forecasting",
        "🧬  Patient Segments",
        "🚨  Risk Engine",
        "🧑‍⚕️  Live Patient Tool",
        "🤖  Clinical Handover",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:.7rem;color:#484f58;'>Dataset: 15,000 ICU Patients<br>ARDS-BUAN305 v2 · 64 features</div>", unsafe_allow_html=True)

# ─── Load data ────────────────────────────────────────────────────────────────
df = load_data()

# ─── Load models (with spinner) ──────────────────────────────────────────────
with st.spinner("Initialising AI models..."):
    mdl = train_models(df)

feat_cols = mdl["feat_cols"]

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("<h1 style='font-size:1.8rem;font-weight:700;color:#e6edf3;margin-bottom:.2rem;'>ICU Overview Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.5rem;'>Real-time population analytics for ARDS patients</p>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns([2,2,1,2])
        age_range = fc1.slider("Age Range", int(df.age.min()), int(df.age.max()), (18, 90))
        sex_sel   = fc2.selectbox("Sex", ["All","Male","Female"])
        sepsis_on = fc3.toggle("Sepsis only", False)
        risk_sel  = fc4.multiselect("Risk Category", ["Low","Medium","High"], default=["Low","Medium","High"])
        st.markdown("</div>", unsafe_allow_html=True)

    dff = df[(df.age >= age_range[0]) & (df.age <= age_range[1])]
    if sex_sel != "All":
        dff = dff[dff.sex == sex_sel]
    if sepsis_on:
        dff = dff[dff.high_risk_comorbidity_flag == 1]
    if risk_sel:
        dff = dff[dff.risk_category.isin(risk_sel)]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1,k2,k3,k4 = st.columns(4)
    def kpi(col, title, val, cls, delta=None):
        d = f"<div class='kpi-delta' style='color:#8b949e;'>{delta}</div>" if delta else ""
        col.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-title'>{title}</div>
          <div class='kpi-value {cls}'>{val}</div>
          {d}
        </div>""", unsafe_allow_html=True)

    kpi(k1,"60-Day Mortality", f"{dff.mortality_60d.mean()*100:.1f}%", "kpi-red", f"{len(dff[dff.mortality_60d==1]):,} patients")
    kpi(k2,"Avg ICU LOS",      f"{dff.icu_los_days.mean():.1f}d",     "kpi-blue", f"Median {dff.icu_los_days.median():.1f}d")
    kpi(k3,"High Risk",        f"{(dff.risk_category=='High').mean()*100:.1f}%","kpi-yellow", f"{(dff.risk_category=='High').sum():,} patients")
    kpi(k4,"Patients",         f"{len(dff):,}", "kpi-green", f"Filtered from {len(df):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(dff, x="age", color="mortality_60d", nbins=30,
            color_discrete_map={0:"#3b82f6",1:"#ef4444"},
            labels={"mortality_60d":"Outcome","age":"Age"},
            title="Age Distribution by Mortality Outcome",
            barmode="overlay", opacity=.75)
        dark_fig(fig)
        fig.update_layout(showlegend=True, legend=dict(title="", orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        risk_cnt = dff.risk_category.value_counts().reset_index()
        risk_cnt.columns = ["Risk","Count"]
        fig = px.pie(risk_cnt, names="Risk", values="Count",
            color="Risk", color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"},
            title="Risk Category Distribution", hole=.55)
        dark_fig(fig)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.box(dff, x="risk_category", y="icu_los_days",
            color="risk_category",
            color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"},
            title="ICU LOS by Risk Category",
            labels={"icu_los_days":"LOS (days)","risk_category":"Risk"})
        dark_fig(fig)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        vitals = dff[["heart_rate_d0","map_d0","respiratory_rate_d0","spo2_d0","sofa_score_d0"]].corr()
        fig = px.imshow(vitals, color_continuous_scale="Blues",
            title="Vitals Correlation (Day 0)",
            labels=dict(color="r"))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Biomarker distributions ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Key Biomarker Distributions</div>", unsafe_allow_html=True)
    bio_cols = ["lactate_d0","crp_d0","sofa_score_d0","pao2_fio2_ratio_d0","creatinine_d0","wbc_d0"]
    bio_labels = ["Lactate D0","CRP D0","SOFA D0","P/F Ratio D0","Creatinine D0","WBC D0"]
    cols = st.columns(3)
    for i, (bc, bl) in enumerate(zip(bio_cols, bio_labels)):
        with cols[i % 3]:
            fig = px.histogram(dff, x=bc, color="mortality_60d", nbins=25,
                color_discrete_map={0:"#3b82f6",1:"#ef4444"}, opacity=.75,
                title=bl, labels={"mortality_60d":"Outcome"}, barmode="overlay")
            dark_fig(fig)
            fig.update_layout(showlegend=False, height=280, margin=dict(l=20,r=10,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MORTALITY INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Mortality" in page:
    st.markdown("<h1 style='font-size:1.8rem;font-weight:700;color:#e6edf3;'>🧠 Mortality Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.5rem;'>Random Forest binary classifier — 60-day mortality prediction</p>", unsafe_allow_html=True)

    # ── Metrics row ──────────────────────────────────────────────────────────
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("AUC-ROC", f"{mdl['mort_auc']:.3f}", "Target ≥ 0.80")
    tn,fp,fn,tp = mdl['mort_cm'].ravel()
    m2.metric("Sensitivity", f"{tp/(tp+fn):.3f}", "True Positive Rate")
    m3.metric("Specificity", f"{tn/(tn+fp):.3f}", "True Negative Rate")
    m4.metric("Precision",   f"{tp/(tp+fp):.3f}", "PPV")

    tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "🔍 Feature Importance", "🎯 SHAP Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(mdl['yte_mort'], mdl['mort_probs'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={mdl['mort_auc']:.3f}",
                line=dict(color="#3b82f6", width=2.5)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance",
                line=dict(color="#484f58", dash="dash")))
            fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", **DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cm = mdl['mort_cm']
            fig = px.imshow(cm, text_auto=True,
                color_continuous_scale=[[0,"#0d1117"],[1,"#3b82f6"]],
                labels=dict(x="Predicted",y="Actual"),
                title="Confusion Matrix",
                x=["Survived","Deceased"], y=["Survived","Deceased"])
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Probability distribution
        prob_df = pd.DataFrame({"prob": mdl['mort_probs'], "actual": mdl['yte_mort']})
        fig = px.histogram(prob_df, x="prob", color="actual", nbins=40,
            color_discrete_map={0:"#10b981",1:"#ef4444"}, opacity=.8, barmode="overlay",
            title="Predicted Probability Distribution", labels={"prob":"Mortality Probability","actual":"Outcome"})
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fi = pd.DataFrame({"Feature": feat_cols,
                           "Importance": mdl['mort_model'].feature_importances_}).sort_values("Importance", ascending=False).head(20)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=[[0,"#1c2128"],[1,"#3b82f6"]],
            title="Top 20 Feature Importances — Mortality Model")
        dark_fig(fig)
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-card'><div class='section-header'>SHAP Global Importance</div>", unsafe_allow_html=True)
        try:
            import shap
            sample_idx = np.random.choice(len(mdl['X_scaled']), 500, replace=False)
            X_sample = mdl['X_scaled'][sample_idx]
            explainer = shap.TreeExplainer(mdl['mort_model'])
            shap_vals = explainer.shap_values(X_sample)
            sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            mean_abs = np.abs(sv).mean(axis=0)
            shap_df = pd.DataFrame({"Feature": feat_cols, "Mean |SHAP|": mean_abs}).sort_values("Mean |SHAP|", ascending=False).head(15)
            fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                color="Mean |SHAP|", color_continuous_scale=[[0,"#1c2128"],[1,"#ef4444"]],
                title="Mean |SHAP| Values — Top 15 Features")
            dark_fig(fig)
            fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP summary: {e}")
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — LOS FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif "LOS" in page:
    st.markdown("<h1 style='font-size:1.8rem;font-weight:700;color:#e6edf3;'>⏱ ICU Length-of-Stay Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.5rem;'>Random Forest regression model — Predicted LOS at admission</p>", unsafe_allow_html=True)

    m1,m2,m3 = st.columns(3)
    m1.metric("RMSE", f"{mdl['los_rmse']:.2f} days")
    m2.metric("MAE",  f"{mdl['los_mae']:.2f} days")
    m3.metric("R²",   f"{1 - mdl['los_rmse']**2 / np.var(mdl['yte_los']):.3f}")

    tab1, tab2 = st.tabs(["📊 Model Evaluation", "🔬 Feature Importance"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            n = min(500, len(mdl['yte_los']))
            idx = np.random.choice(len(mdl['yte_los']), n, replace=False)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mdl['yte_los'][idx], y=mdl['los_pred_vals'][idx],
                mode="markers", marker=dict(color="#3b82f6", opacity=.5, size=5),
                name="Patients"))
            lims = [mdl['yte_los'].min(), mdl['yte_los'].max()]
            fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                line=dict(color="#ef4444", dash="dash"), name="Perfect"))
            fig.update_layout(title="Actual vs Predicted LOS", xaxis_title="Actual (days)",
                yaxis_title="Predicted (days)", **DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            residuals = mdl['yte_los'] - mdl['los_pred_vals']
            fig = px.histogram(residuals, nbins=50, title="Residual Distribution",
                labels={"value":"Residual (days)"},
                color_discrete_sequence=["#10b981"])
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        # LOS distribution
        los_df = pd.DataFrame({"Predicted": mdl['los_pred_vals'], "Actual": mdl['yte_los']})
        fig = px.histogram(los_df.melt(var_name="Type", value_name="LOS"),
            x="LOS", color="Type", nbins=30, opacity=.8, barmode="overlay",
            color_discrete_map={"Actual":"#3b82f6","Predicted":"#f59e0b"},
            title="LOS Distribution — Actual vs Predicted")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fi = pd.DataFrame({"Feature": feat_cols,
                           "Importance": mdl['los_model'].feature_importances_}).sort_values("Importance", ascending=False).head(20)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=[[0,"#1c2128"],[1,"#10b981"]],
            title="Top 20 Feature Importances — LOS Model")
        dark_fig(fig)
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PATIENT SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Segment" in page:
    st.markdown("<h1 style='font-size:1.8rem;font-weight:700;color:#e6edf3;'>🧬 Patient Phenotyping</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.5rem;'>K-Means unsupervised clustering — 3 ARDS phenotypes</p>", unsafe_allow_html=True)

    df2 = df.copy()
    df2["cluster"] = mdl["cluster_labels"]
    cluster_names = {0:"Cluster A — Inflammatory", 1:"Cluster B — Multi-organ", 2:"Cluster C — Mild ARDS"}
    df2["cluster_name"] = df2["cluster"].map(cluster_names)

    tab1, tab2, tab3 = st.tabs(["🗺 PCA Cluster Map", "📊 Cluster Profiles", "🔍 Clinical Insights"])

    with tab1:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(mdl["X_scaled"][:3000])
        pca_df = pd.DataFrame(pca_coords, columns=["PC1","PC2"])
        pca_df["Cluster"] = [cluster_names[c] for c in mdl["cluster_labels"][:3000]]
        pca_df["Mortality"] = df2["mortality_60d"].values[:3000].astype(str)
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
            symbol="Mortality",
            color_discrete_sequence=["#3b82f6","#ef4444","#10b981"],
            title=f"PCA Cluster Plot (n=3000 patients) — Var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%",
            opacity=.65)
        dark_fig(fig)
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        feat_profile = ["age","sofa_score_d0","lactate_d0","crp_d0","pao2_fio2_ratio_d0",
                        "creatinine_d0","albumin_d0","organ_failure_count"]
        profile = df2.groupby("cluster_name")[feat_profile].mean().T
        fig = px.imshow(profile, color_continuous_scale="RdBu_r", aspect="auto",
            title="Cluster Feature Profiles (Mean Values)")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            cnt = df2.cluster_name.value_counts().reset_index()
            cnt.columns = ["Cluster","Count"]
            fig = px.bar(cnt, x="Cluster", y="Count", color="Cluster",
                color_discrete_sequence=["#3b82f6","#ef4444","#10b981"],
                title="Patients per Cluster")
            dark_fig(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            mort_by_clust = df2.groupby("cluster_name")["mortality_60d"].mean().reset_index()
            mort_by_clust.columns = ["Cluster","Mortality Rate"]
            fig = px.bar(mort_by_clust, x="Cluster", y="Mortality Rate", color="Cluster",
                color_discrete_sequence=["#3b82f6","#ef4444","#10b981"],
                title="Mortality Rate by Cluster")
            dark_fig(fig)
            fig.update_layout(showlegend=False, yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        for cid, cname in cluster_names.items():
            subset = df2[df2["cluster"] == cid]
            mort = subset["mortality_60d"].mean()
            los  = subset["icu_los_days"].mean()
            sofa = subset["sofa_score_d0"].mean()
            lact = subset["lactate_d0"].mean()
            icons = ["🔵","🔴","🟢"]
            st.markdown(f"""
            <div class='section-card'>
              <div class='section-header'>{icons[cid]} {cname}</div>
              <div style='display:flex;gap:2rem;flex-wrap:wrap;'>
                <div><span style='color:#8b949e;font-size:.8rem;'>Patients</span><br>
                  <span style='color:#e6edf3;font-weight:600;'>{len(subset):,}</span></div>
                <div><span style='color:#8b949e;font-size:.8rem;'>Mortality</span><br>
                  <span style='color:#ef4444;font-weight:600;'>{mort*100:.1f}%</span></div>
                <div><span style='color:#8b949e;font-size:.8rem;'>Avg LOS</span><br>
                  <span style='color:#3b82f6;font-weight:600;'>{los:.1f}d</span></div>
                <div><span style='color:#8b949e;font-size:.8rem;'>SOFA Score</span><br>
                  <span style='color:#f59e0b;font-weight:600;'>{sofa:.1f}</span></div>
                <div><span style='color:#8b949e;font-size:.8rem;'>Lactate</span><br>
                  <span style='color:#10b981;font-weight:600;'>{lact:.2f}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — RISK ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Risk Engine" in page:
    st.markdown("<h1 style='font-size:1.8rem;font-weight:700;color:#e6edf3;'>🚨 Risk Stratification Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.5rem;'>Gradient Boosting multi-class classifier — Sepsis-Comorbid ARDS risk tiers</p>", unsafe_allow_html=True)

    m1,m2,m3 = st.columns(3)
    m1.metric("Macro AUC-OVR", f"{mdl['risk_auc']:.3f}")
    m2.metric("Risk Classes",  "Low / Medium / High")
    m3.metric("Training Samples", "12,000")

    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔥 Biomarker Analysis", "🗺 Feature Heatmap"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            rc = df.risk_category.value_counts().reset_index()
            rc.columns = ["Risk","Count"]
            fig = px.bar(rc, x="Risk", y="Count", color="Risk",
                color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"},
                title="Risk Category Distribution")
            dark_fig(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            mort_risk = df.groupby("risk_category")["mortality_60d"].mean().reset_index()
            mort_risk.columns = ["Risk","Mortality"]
            fig = px.bar(mort_risk, x="Risk", y="Mortality", color="Risk",
                color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"},
                title="Mortality Rate per Risk Tier",
                text=mort_risk["Mortality"].apply(lambda x: f"{x*100:.1f}%"))
            dark_fig(fig)
            fig.update_layout(showlegend=False, yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

        # Risk by comorbidity
        comorbidities = ["hypertension","diabetes","copd","ckd","cardiovascular_disease","liver_disease"]
        rows = []
        for c in comorbidities:
            for r in ["Low","Medium","High"]:
                frac = df[(df[c]==1)&(df.risk_category==r)].shape[0] / df[df[c]==1].shape[0]
                rows.append({"Comorbidity":c.replace("_"," ").title(),"Risk":r,"Fraction":frac})
        cm_df = pd.DataFrame(rows)
        fig = px.bar(cm_df, x="Comorbidity", y="Fraction", color="Risk", barmode="stack",
            color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"},
            title="Risk Distribution per Comorbidity")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        bio = ["lactate_d0","crp_d0","sofa_score_d0","pao2_fio2_ratio_d0","creatinine_d0","wbc_d0"]
        bio_labels_map = {"lactate_d0":"Lactate","crp_d0":"CRP","sofa_score_d0":"SOFA",
                          "pao2_fio2_ratio_d0":"P/F Ratio","creatinine_d0":"Creatinine","wbc_d0":"WBC"}
        cols = st.columns(2)
        for i, b in enumerate(bio[:4]):
            with cols[i%2]:
                fig = px.violin(df, x="risk_category", y=b, color="risk_category", box=True,
                    color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"},
                    title=f"{bio_labels_map[b]} by Risk Tier", points=False)
                dark_fig(fig)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        grp = df.groupby("risk_category")[bio].mean()
        fig = px.imshow(grp.T, color_continuous_scale="RdYlGn_r", aspect="auto",
            title="Biomarker Heatmap by Risk Category",
            labels={"color":"Mean Value"})
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — LIVE PATIENT TOOL
# ═══════════════════════════════════════════════════════════════════════════════
elif "Live Patient" in page:
    st.markdown("<h1 style='font-size:1.8rem;font-weight:700;color:#e6edf3;'>🧑‍⚕️ Live Patient Assessment Tool</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.5rem;'>Enter patient vitals and labs to receive real-time AI predictions</p>", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Patient Demographics & Comorbidities</div>", unsafe_allow_html=True)
        age   = st.slider("Age", 18, 90, 55)
        sex   = st.selectbox("Sex", ["Male","Female"])
        bmi   = st.slider("BMI", 15.0, 50.0, 26.0)
        smk   = st.selectbox("Smoking Status", ["Never","Former","Current"])
        vent  = st.selectbox("Ventilation Type", ["Non-invasive","Invasive"])
        st.markdown("**Comorbidities**")
        c1c, c2c, c3c = st.columns(3)
        htn  = c1c.checkbox("Hypertension")
        dm   = c2c.checkbox("Diabetes")
        copd = c3c.checkbox("COPD")
        ckd  = c1c.checkbox("CKD")
        cvd  = c2c.checkbox("CVD")
        ld   = c3c.checkbox("Liver Dis.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Day 0 Vitals & Labs</div>", unsafe_allow_html=True)
        hr0  = st.slider("Heart Rate (bpm)", 40, 180, 90)
        map0 = st.slider("MAP (mmHg)", 40, 130, 75)
        rr0  = st.slider("Resp. Rate (/min)", 8, 40, 18)
        spo2_0 = st.slider("SpO₂ (%)", 70, 100, 94)
        pf0  = st.slider("P/F Ratio (Day 0)", 50, 400, 200)
        sofa0 = st.slider("SOFA Score (Day 0)", 0, 24, 8)
        lac0 = st.slider("Lactate D0 (mmol/L)", 0.5, 15.0, 2.0)
        crp0 = st.slider("CRP D0 (mg/L)", 1.0, 300.0, 50.0)
        crea0 = st.slider("Creatinine D0", 0.3, 10.0, 1.2)
        alb0 = st.slider("Albumin D0 (g/dL)", 1.0, 5.0, 3.0)
        plt0 = st.slider("Platelets D0 (×10³)", 20, 500, 180)
        wbc0 = st.slider("WBC D0 (×10³)", 1.0, 30.0, 10.0)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Day 3 Vitals & Labs</div>", unsafe_allow_html=True)
        hr3   = st.slider("Heart Rate D3", 40, 180, 88)
        map3  = st.slider("MAP D3", 40, 130, 72)
        rr3   = st.slider("Resp Rate D3", 8, 40, 17)
        spo2_3 = st.slider("SpO₂ D3 (%)", 70, 100, 93)
        pf3   = st.slider("P/F Ratio D3", 50, 400, 180)
        sofa3 = st.slider("SOFA D3", 0, 24, 7)
        lac3  = st.slider("Lactate D3", 0.5, 15.0, 2.2)
        crp3  = st.slider("CRP D3", 1.0, 300.0, 60.0)
        crea3 = st.slider("Creatinine D3", 0.3, 10.0, 1.3)
        alb3  = st.slider("Albumin D3", 1.0, 5.0, 2.8)
        plt3  = st.slider("Platelets D3", 20, 500, 160)
        wbc3  = st.slider("WBC D3", 1.0, 30.0, 11.0)
        mech_days = st.slider("Mech. Ventilation Days", 0, 29, 5)
        vaso_dur  = st.slider("Vasopressor Duration (days)", 0, 14, 2)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_out:
        # Build feature vector
        sex_enc = 1 if sex == "Male" else 0
        smk_enc = {"Never":2,"Former":0,"Current":1}[smk]
        vent_enc = 1 if vent == "Invasive" else 0
        vaso0 = 1 if lac0 > 4 or sofa0 > 10 else 0
        vaso3 = 1 if lac3 > 4 or sofa3 > 10 else 0
        hrisk = 1 if (ckd or cvd or ld) else 0
        comorb_cnt = sum([htn,dm,copd,ckd,cvd,ld])
        delta_sofa = sofa3 - sofa0
        delta_lac  = lac3 - lac0
        delta_pf   = pf3 - pf0
        delta_crea = crea3 - crea0
        delta_crp  = crp3 - crp0
        shock_idx  = hr0 / max(map0, 1)
        organ_fail = sum([sofa0 >= 4, lac0 >= 4, crea0 >= 3.5])

        # Fill all feat_cols with reasonable defaults then override known ones
        feat_vals = pd.DataFrame([{
            'age':age,'bmi':bmi,'heart_rate_d0':hr0,'map_d0':map0,'respiratory_rate_d0':rr0,'spo2_d0':spo2_0,
            'heart_rate_d3':hr3,'map_d3':map3,'respiratory_rate_d3':rr3,'spo2_d3':spo2_3,
            'pao2_fio2_ratio_d0':pf0,'fio2_d0':0.5,'peep_d0':8.0,'mean_airway_pressure_d0':15.0,
            'pao2_fio2_ratio_d3':pf3,'fio2_d3':0.5,'peep_d3':8.0,'mean_airway_pressure_d3':15.0,'minute_ventilation_d3':8.0,
            'lactate_d0':lac0,'crp_d0':crp0,'albumin_d0':alb0,'platelet_d0':plt0,'bicarbonate_d0':22.0,
            'creatinine_d0':crea0,'bilirubin_d0':1.0,'wbc_d0':wbc0,
            'lactate_d3':lac3,'crp_d3':crp3,'albumin_d3':alb3,'platelet_d3':plt3,'bicarbonate_d3':21.0,
            'creatinine_d3':crea3,'bilirubin_d3':1.2,'wbc_d3':wbc3,
            'sofa_score_d0':sofa0,'sofa_score_d3':sofa3,'delta_sofa':delta_sofa,'delta_lactate':delta_lac,
            'delta_pf_ratio':delta_pf,'delta_creatinine':delta_crea,'delta_crp':delta_crp,
            'shock_index':shock_idx,'organ_failure_count':organ_fail,'comorbidity_count':comorb_cnt,
            'mechanical_ventilation_days':mech_days,'vasopressor_duration':vaso_dur,
            'hypertension':int(htn),'diabetes':int(dm),'copd':int(copd),'ckd':int(ckd),
            'cardiovascular_disease':int(cvd),'liver_disease':int(ld),
            'vasopressor_use_d0':vaso0,'vasopressor_use_d3':vaso3,'high_risk_comorbidity_flag':hrisk,
            'sex_enc':sex_enc,'smoking_status_enc':smk_enc,'ventilation_type_enc':vent_enc,
        }])

        Xp = mdl["scaler"].transform(feat_vals[feat_cols].values)
        mort_prob = mdl["mort_model"].predict_proba(Xp)[0][1]
        los_pred  = float(mdl["los_model"].predict(Xp)[0])
        risk_enc  = mdl["risk_model"].predict(Xp)[0]
        risk_label = mdl["le_risk"].inverse_transform([risk_enc])[0]
        cluster_id = int(mdl["km"].predict(Xp)[0])
        cluster_names_live = {0:"Inflammatory (A)",1:"Multi-organ (B)",2:"Mild ARDS (C)"}

        # Colour helpers
        mort_color = "#ef4444" if mort_prob > .6 else ("#f59e0b" if mort_prob > .35 else "#10b981")
        risk_cls   = {"High":"pred-high","Medium":"pred-medium","Low":"pred-low"}[risk_label]

        # ── Gauge ────────────────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=mort_prob * 100,
            number={"suffix":"%","font":{"color":mort_color,"size":36}},
            delta={"reference":28,"valueformat":".1f","suffix":"% vs avg"},
            title={"text":"60-Day Mortality Risk","font":{"color":"#e6edf3","size":15}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#8b949e","tickwidth":1},
                "bar":{"color":mort_color,"thickness":.25},
                "bgcolor":"#0d1117",
                "steps":[{"range":[0,35],"color":"#10b98120"},
                         {"range":[35,60],"color":"#f59e0b20"},
                         {"range":[60,100],"color":"#ef444420"}],
                "threshold":{"line":{"color":"white","width":2},"thickness":.8,"value":mort_prob*100}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="#161b22", font=dict(family="Inter"), height=250,
            margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Result cards ─────────────────────────────────────────────────────
        r1, r2 = st.columns(2)
        r1.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-title'>ICU Length of Stay</div>
          <div class='kpi-value kpi-blue'>{los_pred:.1f}d</div>
          <div class='kpi-delta' style='color:#8b949e;'>Predicted at admission</div>
        </div>""", unsafe_allow_html=True)
        r2.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-title'>Patient Phenotype</div>
          <div class='kpi-value' style='color:#a78bfa;font-size:1.1rem;margin-top:.4rem;'>{cluster_names_live[cluster_id]}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='section-card'>
          <div class='section-header'>Risk Assessment</div>
          <div style='display:flex;align-items:center;gap:1rem;'>
            <span class='{risk_cls}'>{risk_label} Risk</span>
            <span style='color:#8b949e;font-size:.85rem;'>Sepsis-Comorbid ARDS Tier</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Key drivers ──────────────────────────────────────────────────────
        fi = mdl["mort_model"].feature_importances_
        fi_series = pd.Series(fi, index=feat_cols).sort_values(ascending=False)
        top5 = fi_series.head(5)
        top5_vals = feat_vals[top5.index.tolist()].values[0]

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🔥 Top Driving Factors</div>", unsafe_allow_html=True)
        for fname, fval in zip(top5.index, top5_vals):
            label = fname.replace("_"," ").title()
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:.4rem 0;border-bottom:1px solid #21262d;'>
              <span style='color:#c9d1d9;font-size:.85rem;'>{label}</span>
              <span style='color:#3b82f6;font-weight:600;font-size:.85rem;'>{fval:.2f}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── What-If Simulator ─────────────────────────────────────────────────
        with st.expander("🧪 What-If Simulator"):
            st.markdown("Adjust key parameters to see real-time impact on predictions")
            w1, w2, w3 = st.columns(3)
            wi_spo2  = w1.slider("SpO₂ (sim)", 70, 100, int(spo2_0), key="wi_spo2")
            wi_lac   = w2.slider("Lactate (sim)", 0.5, 15.0, lac0, key="wi_lac")
            wi_sofa  = w3.slider("SOFA (sim)", 0, 24, sofa0, key="wi_sofa")

            feat_sim = feat_vals.copy()
            feat_sim["spo2_d0"] = wi_spo2
            feat_sim["lactate_d0"] = wi_lac
            feat_sim["sofa_score_d0"] = wi_sofa
            Xs = mdl["scaler"].transform(feat_sim[feat_cols].values)
            sim_mort = mdl["mort_model"].predict_proba(Xs)[0][1]
            sim_los  = float(mdl["los_model"].predict(Xs)[0])

            sc1, sc2 = st.columns(2)
            delta_m = sim_mort - mort_prob
            sc1.metric("Simulated Mortality Risk", f"{sim_mort*100:.1f}%",
                f"{delta_m*100:+.1f}% vs current",
                delta_color="inverse")
            delta_l = sim_los - los_pred
            sc2.metric("Simulated LOS", f"{sim_los:.1f}d",
                f"{delta_l:+.1f}d vs current",
                delta_color="inverse")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — CLINICAL HANDOVER
# ═══════════════════════════════════════════════════════════════════════════════
elif "Handover" in page:
    st.markdown("<h1 style='font-size:1.8rem;font-weight:700;color:#e6edf3;'>🤖 AI Clinical Handover</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;margin-bottom:1.5rem;'>Generate structured SBAR / I-PASS handover notes from patient predictions</p>", unsafe_allow_html=True)

    col_form, col_sbar = st.columns([1,1.3])

    with col_form:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Patient Summary Input</div>", unsafe_allow_html=True)
        pt_name    = st.text_input("Patient ID / Name", "ICU-00142")
        pt_age     = st.number_input("Age", 18, 100, 62)
        pt_sex     = st.selectbox("Sex", ["Male","Female"])
        pt_dx      = st.text_input("Primary Diagnosis", "ARDS with Sepsis")
        pt_mort    = st.slider("Predicted Mortality Risk (%)", 0, 100, 45)
        pt_los     = st.slider("Predicted LOS (days)", 1, 29, 14)
        pt_risk    = st.selectbox("Risk Tier", ["High","Medium","Low"])
        pt_cluster = st.selectbox("Phenotype", ["Inflammatory (A)","Multi-organ (B)","Mild ARDS (C)"])
        pt_sofa    = st.slider("SOFA Score", 0, 24, 9)
        pt_pf      = st.slider("P/F Ratio", 50, 400, 160)
        pt_lac     = st.slider("Lactate (mmol/L)", 0.5, 15.0, 3.2)
        pt_crp     = st.slider("CRP (mg/L)", 1.0, 300.0, 85.0)
        pt_vasop   = st.checkbox("Vasopressor use")
        pt_mech    = st.checkbox("Mechanical Ventilation")
        pt_comorbid = st.text_area("Comorbidities", "Hypertension, Type 2 Diabetes")
        pt_current  = st.text_area("Current Interventions", "Invasive mechanical ventilation, norepinephrine drip, prone positioning")
        generate_btn = st.button("🤖 Generate SBAR Note", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sbar:
        if generate_btn:
            mort_tier = "HIGH" if pt_mort >= 60 else ("MODERATE" if pt_mort >= 35 else "LOW")
            mort_color_txt = "#ef4444" if pt_mort >= 60 else ("#f59e0b" if pt_mort >= 35 else "#10b981")
            vent_txt = "currently on mechanical ventilation" if pt_mech else "spontaneously breathing"
            vaso_txt = "vasopressors in use" if pt_vasop else "haemodynamically stable without vasopressors"

            sbar_data = {
                "S — Situation": (
                    f"Patient <strong>{pt_name}</strong>, {pt_age}-year-old {pt_sex}, admitted with "
                    f"<strong>{pt_dx}</strong>. AI risk assessment flags this patient as "
                    f"<span style='color:{mort_color_txt};font-weight:600;'>{mort_tier} RISK</span> "
                    f"with a predicted 60-day mortality of <strong>{pt_mort}%</strong> "
                    f"and estimated ICU LOS of <strong>{pt_los} days</strong>."
                ),
                "B — Background": (
                    f"Relevant comorbidities: <strong>{pt_comorbid}</strong>. "
                    f"Patient is {vent_txt}, with {vaso_txt}. "
                    f"Current interventions include: {pt_current}. "
                    f"Phenotype classification: <strong>{pt_cluster}</strong>."
                ),
                "A — Assessment": (
                    f"Key clinical indicators: SOFA score <strong>{pt_sofa}</strong>, "
                    f"P/F ratio <strong>{pt_pf}</strong> ({"severe ARDS" if pt_pf < 100 else "moderate ARDS" if pt_pf < 200 else "mild ARDS"}), "
                    f"lactate <strong>{pt_lac:.1f} mmol/L</strong> ({"elevated — tissue hypoperfusion" if pt_lac > 2 else "within normal limits"}), "
                    f"CRP <strong>{pt_crp:.0f} mg/L</strong> ({"markedly elevated" if pt_crp > 100 else "elevated"}). "
                    f"Risk stratification: <strong>{pt_risk} tier</strong>. "
                    f"Overall trajectory requires {"urgent escalation of care" if pt_risk == "High" else "close monitoring" if pt_risk == "Medium" else "routine observation"}."
                ),
                "R — Recommendation": (
                    ("• Immediate ICU consultant review and multidisciplinary team escalation. "
                     "• Consider early prone positioning and neuromuscular blockade optimisation. "
                     "• Reassess vasopressor titration per haemodynamic targets. " if pt_risk == "High" else
                     "• Continue current management plan with 6-hourly SOFA reassessment. "
                     "• Optimise fluid balance and ventilator settings. " if pt_risk == "Medium" else
                     "• Maintain current trajectory with daily clinical review. "
                     "• Consider step-down criteria evaluation at 48 hours. ") +
                    f"• Predicted discharge in approximately <strong>{pt_los} days</strong> — coordinate early discharge planning. "
                    f"• Repeat AI risk assessment at Day 3 for trajectory update."
                ),
            }

            st.markdown("<div style='margin-top:.5rem;'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:#161b22;border:1px solid #21262d;border-radius:12px;padding:1.4rem;margin-bottom:1rem;'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;'>
                <div>
                  <div style='font-size:1.1rem;font-weight:700;color:#e6edf3;'>Clinical Handover Note</div>
                  <div style='font-size:.75rem;color:#8b949e;'>Patient: {pt_name} · Generated by ARDS ClinIQ AI</div>
                </div>
                <span class='{"pred-high" if pt_risk=="High" else "pred-medium" if pt_risk=="Medium" else "pred-low"}'>{pt_risk} Risk</span>
              </div>
            """, unsafe_allow_html=True)

            icons = {"S — Situation":"📍","B — Background":"📋","A — Assessment":"⚕️","R — Recommendation":"✅"}
            colors = {"S — Situation":"#3b82f6","B — Background":"#8b5cf6","A — Assessment":"#f59e0b","R — Recommendation":"#10b981"}
            for section, content in sbar_data.items():
                color = colors[section]
                icon  = icons[section]
                st.markdown(f"""
                <div style='background:#1c2128;border-left:3px solid {color};border-radius:0 8px 8px 0;
                            padding:1rem 1.2rem;margin-bottom:.8rem;'>
                  <div style='font-size:.7rem;font-weight:700;letter-spacing:.12em;
                              text-transform:uppercase;color:{color};margin-bottom:.4rem;'>
                    {icon} {section}
                  </div>
                  <div style='font-size:.88rem;color:#c9d1d9;line-height:1.7;'>{content}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.success("✅ SBAR note generated successfully. Copy or print for handover documentation.")

        else:
            st.markdown("""
            <div style='background:#161b22;border:1px dashed #21262d;border-radius:12px;
                        padding:3rem;text-align:center;margin-top:2rem;'>
              <div style='font-size:3rem;margin-bottom:1rem;'>🤖</div>
              <div style='color:#8b949e;font-size:.95rem;line-height:1.7;'>
                Fill in the patient details on the left and click<br>
                <strong style='color:#3b82f6;'>Generate SBAR Note</strong> to create a structured<br>
                clinical handover document powered by AI predictions.
              </div>
            </div>""", unsafe_allow_html=True)
