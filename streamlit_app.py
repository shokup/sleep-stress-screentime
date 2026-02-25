import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Smartphone Usage & Productivity 📱",
    layout="wide",
    page_icon="📱",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {background:#0f0f0f;}
    [data-testid="stSidebar"]          {background:#161616;}
    h1,h2,h3,h4,h5,h6,p,label,div     {color:#f0f0f0!important;}
    .metric-card {
        background:#1e1e1e;border-radius:12px;padding:18px 24px;
        border:1px solid #2a2a2a;margin-bottom:8px;
    }
    .stTabs [data-baseweb="tab"]       {color:#aaa;}
    .stTabs [aria-selected="true"]     {color:#a3e635!important;border-bottom:2px solid #a3e635;}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📱 Smartphone Usage")
page = st.sidebar.selectbox(
    "Navigate",
    ["Introduction 📘", "Visualization 📊", "Insights 🔍", "Prediction 🤖"]
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("smartphone.csv")
    return df

df = load_data()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols     = df.select_dtypes(exclude=np.number).columns.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Introduction
# ─────────────────────────────────────────────────────────────────────────────
if page == "Introduction 📘":
    st.title("📱 Smartphone Usage & Productivity Dashboard")
    st.markdown("Explore how smartphone habits affect productivity, sleep, and stress across 50,000 users.")
    st.write("")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h4>👥 Users</h4><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h4>📵 Avg Daily Hours</h4><h2>{df["Daily_Phone_Hours"].mean():.1f}h</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h4>🧠 Avg Productivity</h4><h2>{df["Work_Productivity_Score"].mean():.1f}/10</h2></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h4>😴 Avg Sleep</h4><h2>{df["Sleep_Hours"].mean():.1f}h</h2></div>', unsafe_allow_html=True)

    st.write("")
    st.subheader("Data Preview")
    rows = st.slider("Rows to display", 5, 30, 10)
    st.dataframe(df.head(rows), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        if missing["Missing"].sum() == 0:
            st.success("✅ No missing values found")
        else:
            st.dataframe(missing[missing["Missing"] > 0])

    with col_b:
        st.subheader("Column Types")
        types = df.dtypes.reset_index()
        types.columns = ["Column", "Type"]
        st.dataframe(types, use_container_width=True)

    if st.button("📊 Show Summary Statistics"):
        st.dataframe(df.describe(), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Visualization
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Visualization 📊":
    st.title("📊 Visualization")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution 📉", "Scatter Plot 🔵", "Category Breakdown 🗂️", "Correlation Heatmap 🔥"
    ])

    with tab1:
        col = st.selectbox("Select numeric column", numeric_cols)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        ax.hist(df[col].dropna(), bins=40, color="#a3e635", edgecolor="#0f0f0f", alpha=0.85)
        ax.set_xlabel(col, color="#ccc"); ax.set_ylabel("Count", color="#ccc")
        ax.tick_params(colors="#ccc"); ax.spines[:].set_color("#333")
        ax.set_title(f"Distribution of {col}", color="#f0f0f0")
        st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X axis", numeric_cols, index=0)
        y_col = c2.selectbox("Y axis", numeric_cols, index=2)
        hue_col = st.selectbox("Color by (optional)", ["None"] + cat_cols)
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        sample = df.sample(min(3000, len(df)))
        if hue_col != "None":
            for val, grp in sample.groupby(hue_col):
                ax.scatter(grp[x_col], grp[y_col], alpha=0.4, s=15, label=str(val))
            ax.legend(labelcolor="#ccc", facecolor="#1e1e1e", edgecolor="#333")
        else:
            ax.scatter(sample[x_col], sample[y_col], color="#a3e635", alpha=0.4, s=15)
        ax.set_xlabel(x_col, color="#ccc"); ax.set_ylabel(y_col, color="#ccc")
        ax.tick_params(colors="#ccc"); ax.spines[:].set_color("#333")
        ax.set_title(f"{x_col} vs {y_col}", color="#f0f0f0")
        st.pyplot(fig)

    with tab3:
        cat = st.selectbox("Select category column", cat_cols)
        metric = st.selectbox("Metric to compare", numeric_cols, index=numeric_cols.index("Work_Productivity_Score") if "Work_Productivity_Score" in numeric_cols else 0)
        agg = df.groupby(cat)[metric].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        bars = ax.bar(agg.index, agg.values, color="#a3e635", edgecolor="#0f0f0f")
        ax.set_xlabel(cat, color="#ccc"); ax.set_ylabel(f"Avg {metric}", color="#ccc")
        ax.tick_params(colors="#ccc"); ax.spines[:].set_color("#333")
        ax.set_title(f"Avg {metric} by {cat}", color="#f0f0f0")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        mask = np.triu(np.ones_like(df[numeric_cols].corr(), dtype=bool))
        sns.heatmap(
            df[numeric_cols].corr(), annot=True, fmt=".2f",
            cmap="RdYlGn", ax=ax, mask=mask,
            linewidths=0.5, linecolor="#0f0f0f",
            annot_kws={"size": 9}
        )
        ax.set_title("Correlation Matrix", color="#f0f0f0", fontsize=14)
        plt.xticks(color="#ccc", rotation=45, ha="right")
        plt.yticks(color="#ccc")
        st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Insights 🔍":
    st.title("🔍 Key Insights")

    # Productivity by occupation & device
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Productivity by Occupation")
        occ = df.groupby("Occupation")["Work_Productivity_Score"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        ax.barh(occ.index, occ.values, color="#a3e635")
        ax.set_xlabel("Avg Productivity", color="#ccc"); ax.tick_params(colors="#ccc")
        ax.spines[:].set_color("#333"); ax.set_title("By Occupation", color="#f0f0f0")
        st.pyplot(fig)

    with c2:
        st.subheader("📱 Device Type Distribution")
        dev = df["Device_Type"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        ax.pie(dev.values, labels=dev.index, colors=["#a3e635","#4ade80","#22d3ee"],
               autopct="%1.1f%%", textprops={"color":"#f0f0f0"})
        st.pyplot(fig)

    st.write("")

    # Stress vs phone hours binned
    st.subheader("😰 Stress Level vs Daily Phone Hours")
    df["Phone_Bin"] = pd.cut(df["Daily_Phone_Hours"], bins=5, precision=1)
    stress_phone = df.groupby("Phone_Bin", observed=True)["Stress_Level"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.bar(stress_phone["Phone_Bin"].astype(str), stress_phone["Stress_Level"], color="#f97316")
    ax.set_xlabel("Daily Phone Hours (binned)", color="#ccc"); ax.set_ylabel("Avg Stress", color="#ccc")
    ax.tick_params(colors="#ccc"); ax.spines[:].set_color("#333")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig)

    # Sleep vs productivity scatter colored by stress
    st.subheader("😴 Sleep Hours vs Productivity (colored by Stress Level)")
    sample = df.sample(min(3000, len(df)))
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    sc = ax.scatter(sample["Sleep_Hours"], sample["Work_Productivity_Score"],
                    c=sample["Stress_Level"], cmap="RdYlGn_r", alpha=0.5, s=15)
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label("Stress Level", color="#ccc")
    cbar.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ccc")
    ax.set_xlabel("Sleep Hours", color="#ccc"); ax.set_ylabel("Productivity Score", color="#ccc")
    ax.tick_params(colors="#ccc"); ax.spines[:].set_color("#333")
    ax.set_title("Sleep vs Productivity", color="#f0f0f0")
    st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Prediction 🤖":
    st.title("🤖 Productivity Prediction")
    st.markdown("Train a model to predict **Work Productivity Score** from usage features.")

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn import metrics

    df2 = df.drop(columns=["User_ID","Phone_Bin"], errors="ignore").copy()
    for col in df2.select_dtypes(exclude=np.number).columns:
        df2[col] = LabelEncoder().fit_transform(df2[col].astype(str))

    all_features = [c for c in df2.columns if c != "Work_Productivity_Score"]
    features_sel = st.sidebar.multiselect("Select Features (X)", all_features, default=all_features)
    model_choice  = st.sidebar.selectbox("Model", ["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting"])
    metrics_sel   = st.sidebar.multiselect("Metrics", ["MSE", "MAE", "R² Score"], default=["MAE", "R² Score"])

    X = df2[features_sel]
    y = df2["Work_Productivity_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    if st.button("🚀 Train Model"):
        with st.spinner(f"Training {model_choice}..."):
            mdl = models[model_choice]
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)

        st.success("Training complete!")
        c1, c2, c3 = st.columns(3)
        if "MSE" in metrics_sel:
            c1.metric("MSE", f"{metrics.mean_squared_error(y_test, preds):.4f}")
        if "MAE" in metrics_sel:
            c2.metric("MAE", f"{metrics.mean_absolute_error(y_test, preds):.4f}")
        if "R² Score" in metrics_sel:
            c3.metric("R²", f"{metrics.r2_score(y_test, preds):.4f}")

        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        ax.scatter(y_test, preds, alpha=0.3, color="#a3e635", s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", lw=2)
        ax.set_xlabel("Actual", color="#ccc"); ax.set_ylabel("Predicted", color="#ccc")
        ax.tick_params(colors="#ccc"); ax.spines[:].set_color("#333")
        ax.set_title("Actual vs Predicted", color="#f0f0f0")
        st.pyplot(fig)

        # Feature Importance (if available)
        if hasattr(mdl, "feature_importances_"):
            st.subheader("🌟 Feature Importance")
            fi = pd.Series(mdl.feature_importances_, index=features_sel).sort_values(ascending=True)
            fig2, ax2 = plt.subplots(figsize=(8, max(4, len(fi)*0.4)), facecolor="#1e1e1e")
            ax2.set_facecolor("#1e1e1e")
            ax2.barh(fi.index, fi.values, color="#a3e635")
            ax2.set_xlabel("Importance", color="#ccc"); ax2.tick_params(colors="#ccc")
            ax2.spines[:].set_color("#333"); ax2.set_title("Feature Importances", color="#f0f0f0")
            st.pyplot(fig2)

        elif hasattr(mdl, "coef_"):
            st.subheader("📐 Coefficients")
            coef = pd.Series(mdl.coef_, index=features_sel).sort_values(ascending=True)
            fig2, ax2 = plt.subplots(figsize=(8, max(4, len(coef)*0.4)), facecolor="#1e1e1e")
            ax2.set_facecolor("#1e1e1e")
            colors = ["#f97316" if v < 0 else "#a3e635" for v in coef.values]
            ax2.barh(coef.index, coef.values, color=colors)
            ax2.axvline(0, color="#ccc", lw=0.8)
            ax2.set_xlabel("Coefficient", color="#ccc"); ax2.tick_params(colors="#ccc")
            ax2.spines[:].set_color("#333"); ax2.set_title("Model Coefficients", color="#f0f0f0")
            st.pyplot(fig2)