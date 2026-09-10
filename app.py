import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

BENIGN_COLOR = "#2ec4b6"
MALIGNANT_COLOR = "#e63946"
ACCENT_COLOR = "#6c5ce7"

st.markdown(f"""
<style>
    .stApp {{
        background-color: #0e1117;
    }}
    .hero {{
        background: linear-gradient(120deg, {ACCENT_COLOR} 0%, #341f97 100%);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
    }}
    .hero h1 {{
        margin: 0;
        font-size: 30px;
    }}
    .hero p {{
        margin: 6px 0 0 0;
        opacity: 0.9;
        font-size: 15px;
    }}
    .result-card {{
        border-radius: 14px;
        padding: 20px 22px;
        text-align: center;
        color: white;
    }}
    .result-card h2 {{
        margin: 0;
        font-size: 24px;
    }}
    .result-card p {{
        margin: 4px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }}
    .benign-card {{ background: linear-gradient(135deg, {BENIGN_COLOR}, #1f9e91); }}
    .malignant-card {{ background: linear-gradient(135deg, {MALIGNANT_COLOR}, #b02a37); }}
    .metric-card {{
        background: #1a1d27;
        border: 1px solid #2d3142;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
    }}
    .metric-card .value {{
        font-size: 22px;
        font-weight: 700;
        color: {ACCENT_COLOR};
    }}
    .metric-card .label {{
        font-size: 12px;
        color: #9aa0b4;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .disclaimer {{
        background: #2a1f14;
        border-left: 4px solid #e8a33d;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        color: #f0d9b5;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    return model, scaler, X, y, data


model, scaler, X, y, data_obj = load_model_and_data()
feature_names = list(data_obj.feature_names)

st.markdown("""
<div class="hero">
    <h1>🩺 Breast Cancer Classifier</h1>
    <p>Random Forest model trained on the Wisconsin Diagnostic Breast Cancer dataset</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["🔮 Predict", "📊 Explore Data", "📈 Model Performance", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="disclaimer">⚠️ Educational demo only — not a medical diagnostic tool.</div>', unsafe_allow_html=True)

FEATURE_GROUPS = {
    "Mean": [f for f in feature_names if f.startswith("mean ")],
    "Standard Error": [f for f in feature_names if f.endswith("error")],
    "Worst": [f for f in feature_names if f.startswith("worst ")],
}

if page == "🔮 Predict":
    st.subheader("Enter Diagnostic Measurements")
    st.caption("Adjust the sliders (grouped by measurement type) or use a preset to auto-fill sample values.")

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    if "preset" not in st.session_state:
        st.session_state.preset = "mean"

    with preset_col1:
        if st.button("Use dataset mean", use_container_width=True):
            st.session_state.preset = "mean"
    with preset_col2:
        if st.button("Typical benign sample", use_container_width=True):
            st.session_state.preset = "benign"
    with preset_col3:
        if st.button("Typical malignant sample", use_container_width=True):
            st.session_state.preset = "malignant"

    if st.session_state.preset == "benign":
        defaults = X[y == 1].mean()
    elif st.session_state.preset == "malignant":
        defaults = X[y == 0].mean()
    else:
        defaults = X.mean()

    user_input = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        with st.expander(f"{group_name} features ({len(group_features)})", expanded=(group_name == "Mean")):
            cols = st.columns(3)
            for idx, feature in enumerate(group_features):
                col = cols[idx % 3]
                with col:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    user_input[feature] = st.slider(
                        feature.replace("mean ", "").replace("worst ", "").replace(" error", " (SE)").title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=float(defaults[feature]),
                        step=(max_val - min_val) / 100,
                        key=feature,
                    )

    input_df = pd.DataFrame([user_input])[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns([1.2, 1, 1])
    with result_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card malignant-card">
                <h2>⚠️ Malignant</h2>
                <p>Likely cancerous tumor</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card benign-card">
                <h2>✓ Benign</h2>
                <p>Non-cancerous tumor</p>
            </div>
            """, unsafe_allow_html=True)
    with result_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{max(probability) * 100:.1f}%</div>
            <div class="label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with result_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{probability[1] * 100:.1f}%</div>
            <div class="label">Malignancy Risk</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        classes = ["Benign", "Malignant"]
        colors = [BENIGN_COLOR, MALIGNANT_COLOR]
        bars = ax.barh(classes, probability, color=colors, edgecolor="none")
        ax.set_xlim([0, 1])
        ax.set_title("Prediction Probability", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, (bar, prob) in enumerate(zip(bars, probability)):
            ax.text(prob + 0.02, i, f"{prob * 100:.1f}%", va="center", color="white", fontweight="bold")
        st.pyplot(fig, transparent=True)

    with viz_col2:
        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.barh(feature_importance["Feature"], feature_importance["Importance"], color=ACCENT_COLOR, edgecolor="none")
        ax.invert_yaxis()
        ax.set_title("Top Features Driving This Model", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig, transparent=True)

elif page == "📊 Explore Data":
    st.subheader("Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Class Comparison"])

    with tab1:
        selected_feature = st.selectbox("Select a feature:", feature_names)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(X[selected_feature], bins=30, edgecolor="white", alpha=0.85, color=ACCENT_COLOR)
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {selected_feature}", fontweight="bold")
            ax.grid(True, alpha=0.2, axis="y")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            data_by_class = [X[selected_feature][y == 0], X[selected_feature][y == 1]]
            bp = ax.boxplot(data_by_class, patch_artist=True)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Malignant", "Benign"])
            for patch, color in zip(bp["boxes"], [MALIGNANT_COLOR, BENIGN_COLOR]):
                patch.set_facecolor(color)
            ax.set_ylabel(selected_feature)
            ax.set_title(f"{selected_feature} by Class", fontweight="bold")
            ax.grid(True, alpha=0.2, axis="y")
            st.pyplot(fig)

    with tab2:
        top_n = st.slider("Number of top features to show:", 5, 15, 10)
        correlation_with_target = X.corrwith(pd.Series(y)).sort_values(ascending=False, key=abs)
        top_features = correlation_with_target.abs().nlargest(top_n).index.tolist()
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = X[top_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title(f"Feature Correlation (Top {top_n})", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        st.pyplot(fig)

        st.subheader("Correlation with Target")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_sorted = correlation_with_target.head(15)
        colors = [MALIGNANT_COLOR if v > 0 else BENIGN_COLOR for v in corr_sorted]
        ax.barh(corr_sorted.index, corr_sorted.values, color=colors, edgecolor="none")
        ax.set_xlabel("Correlation Coefficient")
        ax.set_title("Top Features Correlated with Malignancy", fontweight="bold")
        st.pyplot(fig)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            class_counts = pd.Series(y).value_counts().sort_index()
            ax.bar(["Malignant", "Benign"], class_counts.values, color=[MALIGNANT_COLOR, BENIGN_COLOR], edgecolor="none")
            ax.set_ylabel("Count")
            ax.set_title("Target Distribution", fontweight="bold")
            ax.grid(True, alpha=0.2, axis="y")
            for i, v in enumerate(class_counts.values):
                ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            class_pct = pd.Series(y).value_counts(normalize=True).sort_index() * 100
            ax.pie(class_pct.values, labels=["Malignant", "Benign"], autopct="%1.1f%%", colors=[MALIGNANT_COLOR, BENIGN_COLOR], startangle=90)
            ax.set_title("Class Proportion", fontweight="bold")
            st.pyplot(fig)

        st.subheader("Summary Statistics by Class")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Malignant Tumors**")
            st.dataframe(X[y == 0].describe().T[["mean", "std", "min", "max"]].head(10), use_container_width=True)
        with col2:
            st.write("**Benign Tumors**")
            st.dataframe(X[y == 1].describe().T[["mean", "std", "min", "max"]].head(10), use_container_width=True)

elif page == "📈 Model Performance":
    st.subheader("Model Performance Metrics")

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in zip(
        [m1, m2, m3, m4],
        ["Model Type", "Trees", "Features", "Training Samples"],
        ["Random Forest", "100", "30", str(len(X))],
    ):
        col.markdown(f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

    st.divider()

    X_scaled = scaler.transform(X)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    train_pred = model.predict(X_scaled)

    p1, p2, p3, p4 = st.columns(4)
    for col, label, value in zip(
        [p1, p2, p3, p4],
        ["CV Accuracy (5-fold)", "Training Accuracy", "Precision (Weighted)", "Recall (Weighted)"],
        [
            f"{cv_scores.mean():.1%} \u00b1 {cv_scores.std():.1%}",
            f"{accuracy_score(y, train_pred):.1%}",
            f"{precision_score(y, train_pred, average='weighted'):.1%}",
            f"{recall_score(y, train_pred, average='weighted'):.1%}",
        ],
    ):
        col.markdown(f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, train_pred)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Malignant", "Benign"])
    ax.set_yticklabels(["Malignant", "Benign"])
    ax.set_title("Confusion Matrix (Training Data)", fontweight="bold")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**True Malignant:** {cm[0,0]}  \n**False Benign:** {cm[0,1]}")
    with col2:
        st.write(f"**True Benign:** {cm[1,1]}  \n**False Malignant:** {cm[1,0]}")

elif page == "ℹ️ About":
    st.subheader("About This Application")
    st.markdown("""
This app uses a **Random Forest** classifier to predict whether a breast tumor is benign or malignant, based on 30 diagnostic measurements extracted from digitized images of fine needle aspirates.

**Dataset:** UCI Wisconsin Diagnostic Breast Cancer — 569 samples, 30 features, ~63% benign / ~37% malignant.

**Model:** 100-tree Random Forest on standardized features, evaluated with 5-fold cross-validation.

**Performance:** ~97% accuracy, ~0.99 ROC-AUC.
    """)
    st.markdown('<div class="disclaimer">⚠️ This model is for educational purposes only and is not a substitute for professional medical diagnosis.</div>', unsafe_allow_html=True)

st.divider()
st.caption("Built with Streamlit and scikit-learn")
