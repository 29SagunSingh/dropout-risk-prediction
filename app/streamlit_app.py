import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from recommender import get_recommendations, assign_risk_level

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #f8f7f4; }

.stApp { background-color: #f8f7f4; }

/* Header */
.app-header {
    background: #1a1a2e;
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 600;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #a0a0c0;
    margin: 0;
    font-size: 0.95rem;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card.high   { border-color: #e24b4a; }
.metric-card.medium { border-color: #ef9f27; }
.metric-card.low    { border-color: #1d9e75; }
.metric-card.total  { border-color: #378add; }
.metric-label { font-size: 0.78rem; color: #888; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 2rem; font-weight: 600; line-height: 1.1; }
.metric-card.high   .metric-value { color: #e24b4a; }
.metric-card.medium .metric-value { color: #ef9f27; }
.metric-card.low    .metric-value { color: #1d9e75; }
.metric-card.total  .metric-value { color: #378add; }

/* Risk badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-high   { background: #fdeaea; color: #a32d2d; }
.badge-medium { background: #fef3e2; color: #7a4f10; }
.badge-low    { background: #e6f7f0; color: #0e5c3a; }

/* Student card */
.student-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 5px solid;
}
.student-card.high   { border-color: #e24b4a; }
.student-card.medium { border-color: #ef9f27; }
.student-card.low    { border-color: #1d9e75; }

/* Recommendation card */
.rec-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border-top: 3px solid;
}
.rec-card.academic      { border-color: #378add; }
.rec-card.financial     { border-color: #e24b4a; }
.rec-card.engagement    { border-color: #ef9f27; }
.rec-card.geographic    { border-color: #1d9e75; }
.rec-card.personal      { border-color: #7f77dd; }
.rec-card.socioeconomic { border-color: #d4537e; }

.rec-title { font-weight: 600; font-size: 1rem; margin-bottom: 0.4rem; }
.rec-detail { font-size: 0.88rem; color: #555; line-height: 1.6; }
.rec-meta { font-size: 0.78rem; color: #888; margin-top: 0.5rem; }

/* Risk factor bar */
.factor-bar-container { margin-bottom: 0.6rem; }
.factor-label { font-size: 0.85rem; font-weight: 500; color: #333; margin-bottom: 2px; }
.factor-bar-bg { background: #f0efeb; border-radius: 4px; height: 8px; }
.factor-bar-fill { background: #e24b4a; border-radius: 4px; height: 8px; }

/* Score ring (CSS only) */
.score-ring-container { text-align: center; padding: 1rem 0; }
.score-value { font-size: 3rem; font-weight: 600; }
.score-label { font-size: 0.85rem; color: #888; margin-top: 0.2rem; }

/* Section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a2e;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #eee;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1a2e;
}
section[data-testid="stSidebar"] * {
    color: #c8c8e0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #a0a0c0 !important;
}

div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    model       = joblib.load(os.path.join(base, '..', 'models', 'xgboost_model.pkl'))
    explainer   = joblib.load(os.path.join(base, '..', 'models', 'shap_explainer.pkl'))
    le_dict     = joblib.load(os.path.join(base, '..', 'models', 'label_encoders.pkl'))
    risk_config = joblib.load(os.path.join(base, '..', 'models', 'risk_config.pkl'))
    with open(os.path.join(base, 'intervention_library.json'), 'r') as f:
        library = json.load(f)
    feature_names = pd.read_csv(
        os.path.join(base, '..', 'data', 'processed', 'feature_names.csv'),
        header=None
    )[0].tolist()
    return model, explainer, le_dict, risk_config, library['rules'], feature_names

model, explainer, le_dict, risk_config, rules, FEATURE_NAMES = load_artifacts()


# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_risk(df_input):
    X = df_input[FEATURE_NAMES].copy()
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)
    scores = model.predict_proba(X)[:, 1]
    levels = [assign_risk_level(s) for s in scores]
    return scores, levels


def compute_shap(df_input):
    X = df_input[FEATURE_NAMES].copy()
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)
    return explainer.shap_values(X), X


def risk_color(level):
    return {'High': '#e24b4a', 'Medium': '#ef9f27', 'Low': '#1d9e75'}.get(level, '#888')


def risk_badge(level):
    cls = level.lower()
    return f'<span class="badge badge-{cls}">{level} Risk</span>'


def category_css(cat):
    return cat.lower().replace(' ', '')


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Dropout Risk System")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📤 Upload Data", "📊 Risk Dashboard", "🔍 Student Drilldown", "📋 Recommendations"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#666'>XGBoost + SHAP · UCI Dataset</small>",
        unsafe_allow_html=True
    )

# ── Session state ─────────────────────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = 0


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if "Upload" in page:
    st.markdown("""
    <div class="app-header">
        <h1>🎓 Student Dropout Risk Predictor</h1>
        <p>Upload student data to generate dropout risk scores and personalised intervention recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="section-title">Upload student CSV</div>', unsafe_allow_html=True)
        st.markdown("""
        Your CSV should contain the same columns as the UCI dataset. 
        The system will automatically compute risk scores and SHAP explanations for each student.
        """)

        uploaded = st.file_uploader(
            "Drop your CSV file here",
            type=['csv'],
            help="CSV with student feature columns matching the UCI dataset format"
        )

        st.markdown("---")
        st.markdown("**Or use the built-in processed dataset for a demo:**")
        use_demo = st.button("Load demo dataset", type="primary", use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Required columns (sample)</div>', unsafe_allow_html=True)
        sample_cols = [
            "Age at enrollment", "Gender", "Scholarship holder",
            "Debtor", "Tuition fees up to date",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            "Unemployment rate", "GDP", "..."
        ]
        for col in sample_cols:
            st.markdown(f"- `{col}`")

    if uploaded or use_demo:
        with st.spinner("Loading and processing data..."):
            if uploaded:
                df_raw = pd.read_csv(uploaded)
            else:
                base = os.path.dirname(os.path.abspath(__file__))
                df_raw = pd.read_csv(os.path.join(base, '..', 'data', 'processed', 'full_processed.csv'))

            # Engineer features if raw input (same as preprocessing notebook)
            if 'Approval_Rate_Sem1' not in df_raw.columns:
                df_raw['Approval_Rate_Sem1'] = np.where(
                    df_raw['Curricular units 1st sem (enrolled)'] > 0,
                    df_raw['Curricular units 1st sem (approved)'] / df_raw['Curricular units 1st sem (enrolled)'], 0)
                df_raw['Approval_Rate_Sem2'] = np.where(
                    df_raw['Curricular units 2nd sem (enrolled)'] > 0,
                    df_raw['Curricular units 2nd sem (approved)'] / df_raw['Curricular units 2nd sem (enrolled)'], 0)
                df_raw['Grade_Drop'] = df_raw['Curricular units 1st sem (grade)'] - df_raw['Curricular units 2nd sem (grade)']
                df_raw['Total_Units_Failed'] = (
                    (df_raw['Curricular units 1st sem (enrolled)'] - df_raw['Curricular units 1st sem (approved)']) +
                    (df_raw['Curricular units 2nd sem (enrolled)'] - df_raw['Curricular units 2nd sem (approved)'])
                ).clip(lower=0)
                df_raw['Total_Units_Approved'] = df_raw['Curricular units 1st sem (approved)'] + df_raw['Curricular units 2nd sem (approved)']
                df_raw['Avg_Grade'] = (df_raw['Curricular units 1st sem (grade)'] + df_raw['Curricular units 2nd sem (grade)']) / 2
                df_raw['Approval_Rate_Trend'] = df_raw['Approval_Rate_Sem2'] - df_raw['Approval_Rate_Sem1']
                df_raw['Financial_Risk_Score'] = df_raw['Debtor'].astype(int) + (1 - df_raw['Tuition fees up to date'].astype(int))
                df_raw['Scholarship_Debtor'] = ((df_raw['Scholarship holder'] == 1) & (df_raw['Debtor'] == 1)).astype(int)
                df_raw['Total_Evaluations'] = df_raw['Curricular units 1st sem (evaluations)'] + df_raw['Curricular units 2nd sem (evaluations)']
                df_raw['Enrollment_Drop'] = df_raw['Curricular units 1st sem (enrolled)'] - df_raw['Curricular units 2nd sem (enrolled)']
                df_raw['Mature_Student'] = (df_raw['Age at enrollment'] >= 25).astype(int)

            scores, levels = predict_risk(df_raw)
            shap_vals, X_clean = compute_shap(df_raw)

            results = df_raw.copy()
            results['Risk_Score'] = scores
            results['Risk_Level'] = levels
            results['Student_ID'] = range(1, len(results) + 1)

            st.session_state.df = df_raw
            st.session_state.results_df = results
            st.session_state.shap_values = shap_vals

        st.success(f"✅ Processed {len(results)} students. Navigate to **Risk Dashboard** to view results.")
        st.dataframe(results[['Student_ID', 'Risk_Score', 'Risk_Level']].head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("""
    <div class="app-header">
        <h1>📊 Risk Dashboard</h1>
        <p>Overview of all students ranked by dropout risk score.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.results_df is None:
        st.warning("No data loaded. Please go to **Upload Data** first.")
        st.stop()

    results = st.session_state.results_df

    # ── Summary metrics ──
    high_n   = (results['Risk_Level'] == 'High').sum()
    medium_n = (results['Risk_Level'] == 'Medium').sum()
    low_n    = (results['Risk_Level'] == 'Low').sum()
    total_n  = len(results)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card total">
            <div class="metric-label">Total students</div>
            <div class="metric-value">{total_n}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card high">
            <div class="metric-label">High risk</div>
            <div class="metric-value">{high_n}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card medium">
            <div class="metric-label">Medium risk</div>
            <div class="metric-value">{medium_n}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card low">
            <div class="metric-label">Low risk</div>
            <div class="metric-value">{low_n}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters ──
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        filter_level = st.selectbox("Filter by risk level", ["All", "High", "Medium", "Low"])
    with col_f2:
        sort_by = st.selectbox("Sort by", ["Risk Score ↓", "Risk Score ↑", "Student ID"])
    with col_f3:
        search = st.text_input("Search by Student ID", placeholder="e.g. 42")

    filtered = results.copy()
    if filter_level != "All":
        filtered = filtered[filtered['Risk_Level'] == filter_level]
    if search.strip():
        try:
            filtered = filtered[filtered['Student_ID'] == int(search.strip())]
        except:
            pass
    if sort_by == "Risk Score ↓":
        filtered = filtered.sort_values('Risk_Score', ascending=False)
    elif sort_by == "Risk Score ↑":
        filtered = filtered.sort_values('Risk_Score', ascending=True)
    else:
        filtered = filtered.sort_values('Student_ID')

    st.markdown(f'<div class="section-title">Students — {len(filtered)} shown</div>', unsafe_allow_html=True)

    # ── Student table ──
    display_cols = ['Student_ID', 'Risk_Score', 'Risk_Level',
                    'Age at enrollment', 'Gender',
                    'Curricular units 1st sem (grade)',
                    'Curricular units 2nd sem (grade)',
                    'Tuition fees up to date', 'Debtor']
    available_cols = [c for c in display_cols if c in filtered.columns]
    show_df = filtered[available_cols].copy()
    show_df['Risk_Score'] = show_df['Risk_Score'].round(4)

    st.dataframe(
        show_df.reset_index(drop=True),
        use_container_width=True,
        height=420
    )

    # ── Risk distribution chart ──
    st.markdown('<div class="section-title">Risk score distribution</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor('#f8f7f4')

    axes[0].hist(results[results['Risk_Level'] == 'High']['Risk_Score'],
                 bins=20, color='#e24b4a', alpha=0.75, label='High', edgecolor='white')
    axes[0].hist(results[results['Risk_Level'] == 'Medium']['Risk_Score'],
                 bins=20, color='#ef9f27', alpha=0.75, label='Medium', edgecolor='white')
    axes[0].hist(results[results['Risk_Level'] == 'Low']['Risk_Score'],
                 bins=20, color='#1d9e75', alpha=0.75, label='Low', edgecolor='white')
    axes[0].axvline(0.40, color='#ef9f27', linestyle='--', linewidth=1.2)
    axes[0].axvline(0.70, color='#e24b4a', linestyle='--', linewidth=1.2)
    axes[0].set_xlabel('Risk score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Risk score distribution', fontweight='bold')
    axes[0].legend()
    axes[0].set_facecolor('#f8f7f4')

    level_counts = results['Risk_Level'].value_counts().reindex(['High', 'Medium', 'Low'])
    axes[1].bar(level_counts.index, level_counts.values,
                color=['#e24b4a', '#ef9f27', '#1d9e75'], edgecolor='white', alpha=0.85)
    for i, v in enumerate(level_counts.values):
        axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')
    axes[1].set_title('Students by risk level', fontweight='bold')
    axes[1].set_facecolor('#f8f7f4')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STUDENT DRILLDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif "Drilldown" in page:
    st.markdown("""
    <div class="app-header">
        <h1>🔍 Student Drilldown</h1>
        <p>Detailed risk analysis and SHAP explanation for an individual student.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.results_df is None:
        st.warning("No data loaded. Please go to **Upload Data** first.")
        st.stop()

    results    = st.session_state.results_df
    shap_vals  = st.session_state.shap_values

    student_ids = results['Student_ID'].tolist()
    selected_id = st.selectbox(
        "Select a student",
        student_ids,
        index=int(results['Risk_Score'].idxmax()),
        format_func=lambda x: f"Student {x}  |  Risk: {results[results['Student_ID']==x]['Risk_Score'].values[0]:.3f}"
    )

    idx         = results[results['Student_ID'] == selected_id].index[0]
    row         = results.loc[idx]
    risk_score  = row['Risk_Score']
    risk_level  = row['Risk_Level']
    color       = risk_color(risk_level)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top row ──
    col_score, col_info = st.columns([1, 2])

    with col_score:
        pct = int(risk_score * 100)
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:2rem;text-align:center;
                    box-shadow:0 1px 4px rgba(0,0,0,0.07);border-top:5px solid {color}">
            <div style="font-size:0.8rem;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem">
                Dropout Risk Score
            </div>
            <div style="font-size:3.5rem;font-weight:700;color:{color};line-height:1">{pct}%</div>
            <div style="margin-top:0.8rem">{risk_badge(risk_level)}</div>
            <div style="margin-top:1rem;background:#f0efeb;border-radius:8px;height:10px">
                <div style="width:{pct}%;background:{color};border-radius:8px;height:10px;
                            transition:width 0.5s ease"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-title">Student profile</div>', unsafe_allow_html=True)
        profile_cols = {
            'Age at enrollment'                   : 'Age',
            'Gender'                               : 'Gender',
            'Scholarship holder'                   : 'Scholarship',
            'Debtor'                               : 'Debtor',
            'Tuition fees up to date'             : 'Fees paid',
            'Curricular units 1st sem (grade)'    : 'Grade sem 1',
            'Curricular units 2nd sem (grade)'    : 'Grade sem 2',
            'Approval_Rate_Sem1'                  : 'Approval rate sem 1',
            'Approval_Rate_Sem2'                  : 'Approval rate sem 2',
            'Financial_Risk_Score'                : 'Financial risk',
        }
        items = [(label, row[col]) for col, label in profile_cols.items() if col in row.index]
        half = len(items) // 2
        pc1, pc2 = st.columns(2)
        for label, val in items[:half]:
            pc1.metric(label, round(float(val), 2) if isinstance(val, float) else val)
        for label, val in items[half:]:
            pc2.metric(label, round(float(val), 2) if isinstance(val, float) else val)

    # ── SHAP waterfall ──
    st.markdown('<div class="section-title">SHAP explanation — why this risk score?</div>', unsafe_allow_html=True)

    shap_row    = shap_vals[results.index.get_loc(idx)]
    base_val    = explainer.expected_value
    X_row       = st.session_state.df[FEATURE_NAMES].iloc[results.index.get_loc(idx)]

    explanation = shap.Explanation(
        values=shap_row,
        base_values=base_val,
        data=X_row.values,
        feature_names=FEATURE_NAMES
    )

    fig_shap, ax_shap = plt.subplots(figsize=(11, 6))
    fig_shap.patch.set_facecolor('#f8f7f4')
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig_shap, use_container_width=True)
    plt.close()

    # ── Top risk factors ──
    st.markdown('<div class="section-title">Top risk-driving features</div>', unsafe_allow_html=True)

    shap_series  = pd.Series(shap_row, index=FEATURE_NAMES)
    pos_shap     = shap_series[shap_series > 0].sort_values(ascending=False).head(6)
    max_val      = pos_shap.max() if len(pos_shap) > 0 else 1

    for feat, sv in pos_shap.items():
        fval    = X_row[feat]
        bar_pct = int((sv / max_val) * 100)
        st.markdown(f"""
        <div class="factor-bar-container">
            <div class="factor-label">{feat}
                <span style="color:#888;font-weight:400;font-size:0.8rem"> = {fval:.2f}
                &nbsp;·&nbsp; SHAP: +{sv:.4f}</span>
            </div>
            <div class="factor-bar-bg">
                <div class="factor-bar-fill" style="width:{bar_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Navigate to recommendations ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.session_state.selected_student = selected_id
    if st.button("📋 View recommendations for this student →", type="primary"):
        st.info("Navigate to **Recommendations** in the sidebar.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Recommendations" in page:
    st.markdown("""
    <div class="app-header">
        <h1>📋 Intervention Recommendations</h1>
        <p>Personalised, SHAP-driven action plan for each at-risk student.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.results_df is None:
        st.warning("No data loaded. Please go to **Upload Data** first.")
        st.stop()

    results   = st.session_state.results_df
    shap_vals = st.session_state.shap_values

    # Filter to at-risk students only
    at_risk = results[results['Risk_Level'].isin(['High', 'Medium'])].sort_values(
        'Risk_Score', ascending=False
    )

    col_sel, col_info = st.columns([1.5, 1])
    with col_sel:
        selected_id = st.selectbox(
            "Select at-risk student",
            at_risk['Student_ID'].tolist(),
            format_func=lambda x: (
                f"Student {x}  |  "
                f"{at_risk[at_risk['Student_ID']==x]['Risk_Level'].values[0]} Risk  |  "
                f"{at_risk[at_risk['Student_ID']==x]['Risk_Score'].values[0]:.3f}"
            )
        )
    with col_info:
        row        = at_risk[at_risk['Student_ID'] == selected_id].iloc[0]
        risk_score = row['Risk_Score']
        risk_level = row['Risk_Level']
        color      = risk_color(risk_level)
        st.markdown(f"""
        <div style="background:white;border-radius:10px;padding:1rem 1.5rem;
                    border-left:4px solid {color};margin-top:1.6rem">
            <span style="font-size:1.6rem;font-weight:700;color:{color}">{int(risk_score*100)}%</span>
            <span style="color:#888;margin-left:0.5rem">dropout risk</span>
            &nbsp;&nbsp;{risk_badge(risk_level)}
        </div>
        """, unsafe_allow_html=True)

    # Generate recommendations
    idx       = results[results['Student_ID'] == selected_id].index[0]
    loc_idx   = results.index.get_loc(idx)
    shap_row  = shap_vals[loc_idx]
    X_row     = st.session_state.df[FEATURE_NAMES].iloc[loc_idx]
    shap_dict = dict(zip(FEATURE_NAMES, shap_row))
    feat_dict = X_row.to_dict()

    recs = get_recommendations(feat_dict, shap_dict, FEATURE_NAMES, rules, top_n_factors=6)

    st.markdown(f'<div class="section-title">{len(recs)} recommended interventions</div>',
                unsafe_allow_html=True)

    if not recs:
        st.info("No specific interventions triggered for this student's risk profile.")
    else:
        # Group by category
        categories = {}
        for rec in recs:
            cat = rec['risk_category']
            categories.setdefault(cat, []).append(rec)

        urgency_icon = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        cat_icon     = {
            'Academic': '📚', 'Financial': '💰', 'Engagement': '📈',
            'Geographic': '📍', 'Personal': '🧑', 'Socioeconomic': '🏘️'
        }

        for cat, cat_recs in categories.items():
            st.markdown(
                f"**{cat_icon.get(cat, '')} {cat} interventions**",
                unsafe_allow_html=True
            )
            for rec in cat_recs:
                css_cat = category_css(cat)
                st.markdown(f"""
                <div class="rec-card {css_cat}">
                    <div class="rec-title">{rec['intervention_title']}</div>
                    <div class="rec-detail">{rec['intervention_detail']}</div>
                    <div class="rec-meta">
                        {urgency_icon.get(rec['urgency'], '')} <strong>{rec['urgency']} urgency</strong>
                        &nbsp;·&nbsp; Responsible: <strong>{rec['responsible_party']}</strong>
                        &nbsp;·&nbsp; Triggered by: <code>{rec['trigger_feature']}</code> = {rec['feature_value']:.2f}
                        &nbsp;·&nbsp; Priority score: {rec['priority_score']:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Download report ──
    st.markdown("---")
    report_rows = []
    for rec in recs:
        report_rows.append({
            'Student ID'           : selected_id,
            'Risk Score'           : round(risk_score, 4),
            'Risk Level'           : risk_level,
            'Category'             : rec['risk_category'],
            'Intervention'         : rec['intervention_title'],
            'Detail'               : rec['intervention_detail'],
            'Urgency'              : rec['urgency'],
            'Responsible Party'    : rec['responsible_party'],
            'Trigger Feature'      : rec['trigger_feature'],
            'Feature Value'        : rec['feature_value'],
            'SHAP Value'           : rec['shap_value'],
            'Priority Score'       : rec['priority_score']
        })

    report_df = pd.DataFrame(report_rows)
    csv_data  = report_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Download recommendation report (CSV)",
        data=csv_data,
        file_name=f"student_{selected_id}_recommendations.csv",
        mime="text/csv",
        use_container_width=True
    )
