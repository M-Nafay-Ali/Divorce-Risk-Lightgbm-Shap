import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt

# --- 1. Page Configuration & Custom CSS Styling ---
st.set_page_config(
    page_title="Marital Dynamics & Divorce Risk Engine",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Background Styling via Glassmorphism and Unsplash Couples Image
custom_css = """
<style>
.stApp {
    background: url("https://images.unsplash.com/photo-1515934751635-c81c6bc9a2d8?q=80&w=1920&auto=format&fit=crop") no-repeat center center fixed;
    background-size: cover;
}

/* Semi-transparent container card effect */
div[data-testid="stVerticalBlock"] > div {
    background: rgba(15, 23, 42, 0.82);
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: rgba(10, 15, 30, 0.9) !important;
}

h1, h2, h3, h4, label, p, .stMarkdown {
    color: #F8FAFC !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 2. Load Pipeline Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model_pipeline.pkl")
        X_test = pd.read_csv("preprocessed_X_test.csv")
        return model, X_test
    except Exception:
        return None, None

model_pipeline, preprocessed_X_test = load_artifacts()

# Title Header
st.title("💍 Marital Dynamics & Relationship Risk Intelligence")
st.caption("Powered by Regularized LightGBM & TreeSHAP Interpretability Mechanics")

if model_pipeline is None:
    st.error("⚠️ Artifact `model_pipeline.pkl` or `preprocessed_X_test.csv` not found! Please run the notebook export cell first.")
    st.stop()

# --- 3. Interactive Sidebar Controls ---
st.sidebar.header("🕹️ Couple Assessment Inputs")

st.sidebar.subheader("Demographics & History")
marriage_number = st.sidebar.slider("Marriage Number", 1, 3, 1)
age_at_marriage = st.sidebar.slider("Age at Marriage", 18, 65, 28)
education_level = st.sidebar.selectbox("Education Level", ["less_than_hs", "high_school", "some_college", "bachelors", "graduate"], index=3)
religious_attendance = st.sidebar.selectbox("Religious Attendance", ["never", "rarely", "monthly", "weekly"], index=1)
financial_stress = st.sidebar.slider("Financial Stress (0 - 10)", 0.0, 10.0, 3.5, 0.1)

st.sidebar.subheader("Gottman Behavioral Indicators")
contempt = st.sidebar.slider("Contempt Score", 0.0, 10.0, 2.1, 0.1)
criticism = st.sidebar.slider("Criticism Score", 0.0, 10.0, 3.0, 0.1)
defensiveness = st.sidebar.slider("Defensiveness Score", 0.0, 10.0, 2.8, 0.1)
stonewalling = st.sidebar.slider("Stonewalling Score", 0.0, 10.0, 2.0, 0.1)

st.sidebar.subheader("De-escalation & Interaction Metrics")
repair_attempt_success = st.sidebar.slider("Repair Attempt Success (0 - 10)", 0.0, 10.0, 6.5, 0.1)
positive_negative_ratio = st.sidebar.slider("Positive-to-Negative Interaction Ratio", 0.1, 20.0, 5.0, 0.1)

# Default non-slider features set to reasonable baseline constants
input_dict = {
    "marriage_number": marriage_number,
    "age_at_marriage": age_at_marriage,
    "age_gap_years": 2,
    "education_level": education_level,
    "household_income_usd": 75000.0,
    "financial_stress": financial_stress,
    "both_employed": 1,
    "cohabited_before": 1,
    "premarital_counseling": 0,
    "child_before_marriage": 0,
    "n_children": 1,
    "religious_attendance": religious_attendance,
    "criticism": criticism,
    "contempt": contempt,
    "defensiveness": defensiveness,
    "stonewalling": stonewalling,
    "repair_attempt_success": repair_attempt_success,
    "positive_negative_ratio": positive_negative_ratio,
    "conflict_frequency_weekly": 2.0,
    "shared_activities_weekly": 3.0
}

input_df = pd.DataFrame([input_dict])

# --- 4. Main Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Prediction Risk Gauge", "🔍 Explainability (SHAP)", "📈 Global Behavioral Trends"])

with tab1:
    st.subheader("Predictive Risk Assessment")
    
    # Generate model probabilities
    prob_divorce = model_pipeline.predict_proba(input_df)[0][1]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Radial Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_divorce * 100,
            number={'suffix': "%", 'font': {'color': 'white', 'size': 40}},
            title={'text': "Predicted Divorce Risk", 'font': {'size': 20, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': "#EF4444" if prob_divorce > 0.5 else "#10B981"},
                'steps': [
                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.3)"},
                    {'range': [35, 65], 'color': "rgba(245, 158, 11, 0.3)"},
                    {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                ],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col2:
        st.markdown("### Risk Level Verdict")
        if prob_divorce > 0.65:
            st.error("🚨 **High Relationship Risk Detected**")
            st.write("Elevated Gottman behavioral markers (contempt/criticism) combined with financial stress push the predicted outcome into a high-risk tier.")
        elif prob_divorce > 0.35:
            st.warning("⚠️ **Moderate Relationship Vulnerability**")
            st.write("Balanced positive and negative attributes. Enhancing repair attempt frequency could help protect stability long term.")
        else:
            st.success("🛡️ **Strong Relationship Stability**")
            st.write("High interaction ratios and effective repair attempts act as strong protective buffers against marital dissolution.")

with tab2:
    st.subheader("Local Model Explainability (TreeSHAP Waterfall)")
    st.caption("Understand how every individual factor pulls the prediction away from the baseline probability.")
    
    try:
        # Preprocess single instance through ColumnTransformer step
        transformed_instance = model_pipeline.named_steps["preprocessor"].transform(input_df)
        fitted_classifier = model_pipeline.named_steps["classifier"]
        
        explainer = shap.TreeExplainer(fitted_classifier)
        shap_vals = explainer(transformed_instance)
        
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        shap.plots.waterfall(shap_vals[0], show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.info("Run full SHAP explainer computation directly from notebook outputs.")

with tab3:
    st.subheader("Interactive Feature Distribution Explorer")
    
    # Plotly Scatter Interaction between Contempt and Positive-Negative Ratio
    fig_scatter = px.scatter(
        preprocessed_X_test,
        x="contempt",
        y="positive_negative_ratio",
        color="education_level",
        title="Interactive Gottman Dynamics: Contempt vs Interaction Ratio",
        template="plotly_dark"
    )
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)

