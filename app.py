import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="wide")

@st.cache_resource
def load():
    d = Path(__file__).parent / "model"
    return joblib.load(d / "pipeline.joblib"), json.loads((d / "meta.json").read_text())

try:
    model, meta = load()
except FileNotFoundError:
    st.error("Model not found. Run  `python train.py \"CROP YIELD.csv\"`  first.")
    st.stop()

cats, dflt, rng = meta['categories'], meta['defaults'], meta['ranges']
def level(v):
    return ("Low Yield" if v < meta['low_threshold']
            else "Medium Yield" if v <= meta['high_threshold'] else "High Yield")

def num(label, col, step=1.0, container=st.sidebar):
    lo, hi = rng[col]
    return container.number_input(label, float(lo), float(hi * 2), float(dflt[col]), step)

st.title("🌾 Crop Yield Predictor")
st.caption(f"Random Forest trained on {meta['n_rows']:,} records · "
           f"R² {meta['metrics']['R2']:.3f} · MAE {meta['metrics']['MAE']:.0f} kg/ha")

sb = st.sidebar
sb.header("Inputs")
inp = {
    'District': sb.selectbox("District", cats['District']),
    'Crop': sb.selectbox("Crop", cats['Crop']),
    'Season': sb.selectbox("Season", cats['Season']),
    'Year': sb.number_input("Year", 1990, 2100, int(dflt['Year'])),
    'Area_ha': num("Area (ha)", 'Area_ha', 10.0),
    'Soil_Type': sb.selectbox("Soil type", cats['Soil_Type']),
    'Soil_pH': sb.slider("Soil pH", 4.0, 9.0, float(np.clip(dflt['Soil_pH'], 4, 9)), 0.1),
    'Soil_Moisture_pct': sb.slider("Soil moisture (%)", 0.0, 60.0, float(np.clip(dflt['Soil_Moisture_pct'], 0, 60)), 1.0),
    'Annual_Rainfall_mm': num("Annual rainfall (mm)", 'Annual_Rainfall_mm', 50.0),
    'Seasonal_Rainfall_mm': num("Seasonal rainfall (mm)", 'Seasonal_Rainfall_mm', 25.0),
    'Mean_Temperature_C': num("Mean temperature (°C)", 'Mean_Temperature_C', 0.5),
    'Relative_Humidity_pct': num("Relative humidity (%)", 'Relative_Humidity_pct', 1.0),
    'Nitrogen_kg_ha': num("Nitrogen (kg/ha)", 'Nitrogen_kg_ha', 5.0),
    'Phosphorus_kg_ha': num("Phosphorus (kg/ha)", 'Phosphorus_kg_ha', 5.0),
    'Potassium_kg_ha': num("Potassium (kg/ha)", 'Potassium_kg_ha', 5.0),
    'Irrigation_Type': sb.selectbox("Irrigation", cats['Irrigation_Type']),
}
inp['Fertilizer_Total_NPK_kg_ha'] = inp['Nitrogen_kg_ha'] + inp['Phosphorus_kg_ha'] + inp['Potassium_kg_ha']
sb.caption(f"Total NPK = {inp['Fertilizer_Total_NPK_kg_ha']:.0f} kg/ha (N+P+K)")

COLS = list(model.feature_names_in_)
def predict(rows):
    return model.predict(pd.DataFrame(rows)[COLS])

t1, t2, t3 = st.tabs(["Prediction", "What-if analysis", "Model details"])

with t1:
    p = float(predict([inp])[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted yield", f"{p:,.0f} kg/ha")
    c2.metric("Yield level", level(p))
    c3.metric("Typical error (MAE)", f"± {meta['metrics']['MAE']:,.0f} kg/ha")
    st.info(f"Low < {meta['low_threshold']:,.0f} · Medium up to {meta['high_threshold']:,.0f} · "
            f"High above that (kg/ha, based on training-data terciles).")

with t2:
    st.write("Vary one factor while everything else stays as set in the sidebar.")
    def sweep(col, values, extra=None):
        rows = []
        for v in values:
            r = dict(inp); r[col] = v
            if extra: r.update(extra(v))
            rows.append(r)
        return pd.DataFrame({col: values, 'Predicted yield (kg/ha)': predict(rows)}).set_index(col)

    a, b = st.columns(2)
    lo, hi = rng['Annual_Rainfall_mm']
    a.subheader("Annual rainfall")
    a.line_chart(sweep('Annual_Rainfall_mm', list(np.linspace(lo, hi, 20)),
                       lambda v: {'Seasonal_Rainfall_mm': v * inp['Seasonal_Rainfall_mm'] / max(inp['Annual_Rainfall_mm'], 1)}))
    b.subheader("Total NPK fertilizer")
    ft = np.linspace(20, max(220, inp['Fertilizer_Total_NPK_kg_ha'] * 1.5), 20)
    s = max(inp['Fertilizer_Total_NPK_kg_ha'], 1)
    b.line_chart(sweep('Fertilizer_Total_NPK_kg_ha', list(ft), lambda v: {
        'Nitrogen_kg_ha': v * inp['Nitrogen_kg_ha'] / s, 'Phosphorus_kg_ha': v * inp['Phosphorus_kg_ha'] / s,
        'Potassium_kg_ha': v * inp['Potassium_kg_ha'] / s}))
    c, d = st.columns(2)
    c.subheader("Soil moisture (%)")
    c.line_chart(sweep('Soil_Moisture_pct', list(np.linspace(5, 50, 20))))
    d.subheader("Irrigation type")
    d.bar_chart(sweep('Irrigation_Type', cats['Irrigation_Type']))
    st.subheader("Crop comparison under these conditions")
    st.bar_chart(sweep('Crop', cats['Crop']))

with t3:
    m = meta['metrics']
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{m['R2']:.3f}"); c2.metric("MAE", f"{m['MAE']:,.0f} kg/ha"); c3.metric("RMSE", f"{m['RMSE']:,.0f} kg/ha")
    st.subheader("Top feature importances")
    st.bar_chart(pd.Series(meta['importances']).sort_values())
    st.caption(f"Best parameters: {meta['best_params']}")
