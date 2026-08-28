import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="PriceMind AI", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0b1120; color: #f1f5f9; font-family: 'Inter', sans-serif; }
    
    /* Remove outline on tabs */
    button[data-baseweb="tab"]:focus { outline: none !important; box-shadow: none !important; }
    
    .pm-topbar { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; background: #111827; border-bottom: 1px solid #1f2937; border-radius: 8px; margin-bottom: 1.5rem; }
    .pm-brand { display: flex; align-items: center; gap: 12px; }
    .pm-logo { background: linear-gradient(135deg, #5eead4, #3b82f6); color: #0b1120; font-weight: 800; font-size: 1.25rem; padding: 8px 12px; border-radius: 6px; }
    .pm-title { font-size: 1.2rem; font-weight: 700; color: #f8fafc; }
    .pm-subtitle { font-size: 0.8rem; color: #94a3b8; }
    .pm-status { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #5eead4; background: rgba(94, 234, 212, 0.1); padding: 4px 10px; border-radius: 20px; }
    .pm-dot { width: 8px; height: 8px; background-color: #5eead4; border-radius: 50%; }
    .pm-kicker { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #38bdf8; font-weight: 600; margin-bottom: 4px; }
    .pm-section-title { font-size: 1.5rem; font-weight: 700; color: #f8fafc; }
    
    /* Separated Metric Cards Layout */
    .pm-metric-container { display: flex; gap: 1rem; margin-top: 1rem; margin-bottom: 1rem; }
    .pm-metric-card { background-color: #1e293b; border: 1px solid #334155; padding: 1.5rem; border-radius: 10px; flex: 1; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    
    .pm-metric-label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px; }
    .pm-metric-val { font-size: 1.5rem; font-weight: 700; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .pm-footer { text-align: center; padding: 2rem 0; color: #64748b; font-size: 0.8rem; border-top: 1px solid #1e293b; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

SEASONALITY_MAPPING = {
    "Normal Season": 1.00,
    "Summer Peak": 1.08,
    "Winter / Holiday Season": 1.12,
    "Back to School": 1.05,
    "Black Friday / Mega Sale": 1.25
}

PROMO_MAPPING = {
    "0% (No Promo)": 0.00,
    "5% Off": 0.05,
    "10% Off": 0.10,
    "15% Off": 0.15,
    "20% Off": 0.20,
    "25% Off": 0.25,
    "30% Off": 0.30
}

def format_currency(val):
    return f"${val:,.2f}"

def format_pct(val, decimals=1):
    return f"{val * 100:.{decimals}f}%"

def get_risk_html(margin, comp_idx):
    if margin < 0.15 or comp_idx > 1.2:
        return '<span style="color: #fb7185; font-weight: bold;">High Risk</span>'
    elif margin < 0.30:
        return '<span style="color: #fbbf24; font-weight: bold;">Moderate Risk</span>'
    return '<span style="color: #34d399; font-weight: bold;">Low Risk (Optimal)</span>'

@st.cache_data
def generate_data(n=2400):
    np.random.seed(42)
    categories = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports"]
    channels = ["Web", "Mobile App", "Direct Sales", "Partner"]
    segments = ["Mass Market", "Budget Shoppers", "Middle Class", "Premium / Luxury", "B2B / Wholesale"]
    
    df = pd.DataFrame({
        "Product_ID": [f"SKU-{np.random.randint(10000, 99999)}" for _ in range(n)],
        "Category": np.random.choice(categories, n),
        "Channel": np.random.choice(channels, n),
        "Customer_Segment": np.random.choice(segments, n),
        "Current_Price": np.random.uniform(10, 500, n).round(2),
        "Unit_Cost": np.random.uniform(5, 250, n).round(2),
        "Competitor_Price": np.random.uniform(10, 500, n).round(2),
        "Inventory": np.random.randint(10, 1000, n),
        "Traffic": np.random.randint(1000, 50000, n),
        "Promotion": np.random.choice(list(PROMO_MAPPING.keys()), n),
        "Seasonality": np.random.choice(list(SEASONALITY_MAPPING.keys()), n)
    })
    
    df.loc[df['Current_Price'] < df['Unit_Cost'], 'Current_Price'] = df['Unit_Cost'] * np.random.uniform(1.1, 1.5)
    
    s_num = df["Seasonality"].map(SEASONALITY_MAPPING)
    p_num = df["Promotion"].map(PROMO_MAPPING)
    
    base_cr = 0.03 
    price_adv = df["Competitor_Price"] / df["Current_Price"]
    price_adv = np.clip(price_adv, 0.2, 3.0)
    
    cr = base_cr * price_adv * s_num * (1 + (p_num * 1.5))
    cr = np.clip(cr, 0.0, 1.0)
    
    demand = (df["Traffic"] * cr).astype(int)
    units_sold = np.minimum(demand, df["Inventory"])
    
    df["Revenue"] = (units_sold * df["Current_Price"]).round(2)
    return df

def get_target(df):
    for c in ["Revenue", "Sales", "Total_Sales"]:
        if c in df.columns: return c
    return None

def build_model(df, target):
    d = df.copy()
    if "Seasonality" in d.columns: d["Seasonality"] = d["Seasonality"].map(SEASONALITY_MAPPING).fillna(1.0)
    if "Promotion" in d.columns: d["Promotion"] = d["Promotion"].map(PROMO_MAPPING).fillna(0.0)
        
    features = [c for c in d.select_dtypes(include=[np.number]).columns if c != target]
    X, y = d[features], d[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    
    return {
        "model": rf, "features": features, "target": target,
        "r2": r2_score(y_test, preds), "rmse": np.sqrt(mean_squared_error(y_test, preds)), "mae": mean_absolute_error(y_test, preds),
        "test_n": len(y_test), "y_test": y_test, "preds": preds,
        "importances": pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_}).sort_values("Importance")
    }

def run_app():
    st.markdown('<div class="pm-topbar"><div class="pm-brand"><div class="pm-logo">PM</div><div><div class="pm-title">PriceMind AI</div><div class="pm-subtitle">Enterprise Pricing & Revenue Intelligence Engine</div></div></div><div class="pm-status"><div class="pm-dot"></div> System Online</div></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Data Configuration")
        src = st.radio("Source Mode", ["Generate Synthetic Data", "Upload Dataset"])
        df = None
        
        if src == "Upload Dataset":
            file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
            if file:
                df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        else:
            if "df" not in st.session_state or st.button("Regenerate Data"):
                st.session_state["df"] = generate_data()
            df = st.session_state["df"]

        st.markdown('<div class="pm-footer">PriceMind AI © 2026</div>', unsafe_allow_html=True)

    if df is None or df.empty:
        st.info("Awaiting data input.")
        return

    target = get_target(df)
    if not target:
        st.error("Target column not found.")
        return

    with st.spinner("Training engine..."):
        m_info = build_model(df, target)

    t1, t2, t3 = st.tabs(["Model Dashboard", "Single Price Simulator", "Batch Scoring Engine"])

    with t1:
        st.markdown('<div class="pm-kicker">OVERVIEW</div><div class="pm-section-title">Engine Performance</div><br>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        
        # Added explanatory tooltips to metrics using the 'help' argument
        m1.metric(
            label="Model R² Score", 
            value=format_pct(m_info["r2"], 2),
            help="Indicates how well the model predicts revenue variation. 100% means perfect accuracy."
        )
        m2.metric(
            label="RMSE", 
            value=format_currency(m_info["rmse"]),
            help="Root Mean Squared Error: Measures prediction standard error in dollars. Penalizes large errors heavily."
        )
        m3.metric(
            label="MAE", 
            value=format_currency(m_info["mae"]),
            help="Mean Absolute Error: The average dollar difference between model predictions and actual values."
        )
        m4.metric(
            label="Test Set Size", 
            value=f"{m_info['test_n']:,} rows",
            help="Number of unseen rows used to validate model accuracy."
        )

        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            fig1 = px.scatter(x=m_info["y_test"], y=m_info["preds"], labels={"x": "Actual", "y": "Predicted"}, title="Convergence", template="plotly_dark", color_discrete_sequence=["#5eead4"], opacity=0.5)
            fig1.update_traces(hovertemplate='<b>Actual:</b> %{x:,.2f}<br><b>Predicted:</b> %{y:,.2f}<extra></extra>')
            mx = max(m_info["y_test"].max(), m_info["preds"].max())
            fig1.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect Fit", line=dict(color="#fb7185", dash="dash"), hoverinfo="skip"))
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            fig2 = px.bar(m_info["importances"], x="Importance", y="Feature", orientation="h", title="Feature Impact", template="plotly_dark", color_discrete_sequence=["#38bdf8"])
            fig2.update_traces(hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>')
            st.plotly_chart(fig2, use_container_width=True)

    with t2:
        st.markdown('<div class="pm-kicker">SCENARIO PLANNING</div><div class="pm-section-title">Single SKU Simulator</div><br>', unsafe_allow_html=True)
        
        with st.form("sim_form"):
            c1, c2, c3, c4 = st.columns(4)
            p_val = c1.number_input("Target Price ($)", value=float(df["Current_Price"].mean()), min_value=1.0, step=1.0)
            c_val = c2.number_input("Unit Cost ($)", value=float(df["Unit_Cost"].mean()), min_value=0.0)
            comp_val = c3.number_input("Competitor Price ($)", value=float(df["Competitor_Price"].mean()), min_value=0.0)
            inv_val = c4.number_input("Inventory Level (Units)", value=int(df["Inventory"].mean()), min_value=0, step=1, format="%d")

            traf_val = c1.number_input("Est. Traffic", value=int(df["Traffic"].mean()), min_value=0, step=1, format="%d")
            pro_val = c2.selectbox("Promotion Level", list(PROMO_MAPPING.keys()))
            seas_val = c3.selectbox("Seasonality Event", list(SEASONALITY_MAPPING.keys()))
            cat_val = c4.selectbox("Category", df["Category"].unique() if "Category" in df.columns else ["Generic"])

            chan_val = c1.selectbox("Channel", df["Channel"].unique() if "Channel" in df.columns else ["Web"])
            seg_val = c2.selectbox("Customer Segment", ["Mass Market", "Budget Shoppers", "Middle Class", "Premium / Luxury", "B2B / Wholesale"])
            
            st.write("")
            submit = st.form_submit_button("Execute Simulation", type="primary")

        if submit:
            row = df.iloc[0:1].copy()
            if "Current_Price" in row: row["Current_Price"] = p_val
            if "Unit_Cost" in row: row["Unit_Cost"] = c_val
            if "Competitor_Price" in row: row["Competitor_Price"] = comp_val
            if "Inventory" in row: row["Inventory"] = inv_val
            if "Traffic" in row: row["Traffic"] = traf_val
            if "Promotion" in row: row["Promotion"] = PROMO_MAPPING.get(pro_val, 0.0)
            if "Seasonality" in row: row["Seasonality"] = SEASONALITY_MAPPING.get(seas_val, 1.0)
            if "Category" in row: row["Category"] = cat_val
            if "Channel" in row: row["Channel"] = chan_val
            if "Customer_Segment" in row: row["Customer_Segment"] = seg_val
            
            pred = max(m_info["model"].predict(row[m_info["features"]])[0], 0.0)
            margin = (p_val - c_val) / p_val if p_val > 0 else 0
            idx = p_val / comp_val if comp_val > 0 else 1.0
            
            st.markdown(f'''
            <div class="pm-metric-container">
                <div class="pm-metric-card">
                    <div class="pm-metric-label">Projected {target}</div>
                    <div class="pm-metric-val">{format_currency(pred)}</div>
                </div>
                <div class="pm-metric-card">
                    <div class="pm-metric-label">Gross Margin</div>
                    <div class="pm-metric-val">{format_pct(margin)}</div>
                </div>
                <div class="pm-metric-card">
                    <div class="pm-metric-label">Competitor Index</div>
                    <div class="pm-metric-val">{idx:.2f}x</div>
                </div>
                <div class="pm-metric-card">
                    <div class="pm-metric-label">Risk Analysis</div>
                    <div class="pm-metric-val">{get_risk_html(margin, idx)}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="pm-kicker">AT-SCALE INTELLIGENCE</div><div class="pm-section-title">Batch Scoring Pipeline</div><br>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True)

        if st.button("Score Entire Dataset", type="primary", use_container_width=True):
            res = df.copy()
            if "Seasonality" in res: res["Seasonality"] = res["Seasonality"].map(SEASONALITY_MAPPING).fillna(1.0)
            if "Promotion" in res: res["Promotion"] = res["Promotion"].map(PROMO_MAPPING).fillna(0.0)
            
            res[f"Predicted_{target}"] = np.maximum(m_info["model"].predict(res[m_info["features"]]), 0.0)
            
            cols = [c for c in ["Product_ID", "Current_Price", target, f"Predicted_{target}"] if c in res.columns]
            st.dataframe(res[cols].head(500), use_container_width=True)
            st.download_button("📥 Download Scored Dataset (CSV)", res.to_csv(index=False).encode('utf-8'), "Scored_Data.csv", "text/csv", use_container_width=True)

if __name__ == "__main__":
    run_app()