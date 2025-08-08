# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")
st.title("🧠 Psychiatric Dropout Risk Predictor")

# 載入
model = joblib.load("dropout_model.pkl")
sample = pd.read_csv("sample_input.csv")
with open("thresholds.json","r") as f:
    THR = json.load(f)
HIGH_CUT  = THR["high_cut"]
YOUDEN_CUT = THR["youden_cut"]

# 側欄輸入
with st.sidebar:
    st.header("Patient Info")
    age = st.slider("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male","Female"])
    diagnosis = st.selectbox("Diagnosis", [
        "Schizophrenia","Bipolar","Depression",
        "Personality Disorder","Substance Use Disorder","Dementia"
    ])
    length_of_stay = st.slider("Length of Stay (days)", 1, 90, 10)
    num_adm = st.slider("# Previous Admissions", 0, 15, 1)
    social_worker = st.radio("Has Social Worker", ["Yes","No"])
    compliance = st.slider("Medication Compliance Score", 0.0, 10.0, 5.0, 0.1)
    recent_self_harm = st.radio("Recent Self-harm", ["Yes","No"])
    selfharm_adm = st.radio("Self-harm During Admission", ["Yes","No"])
    support = st.slider("Family Support Score", 0.0, 10.0, 5.0, 0.1)
    followups = st.slider("Post-discharge Followups", 0, 10, 2)

# 單筆 dataframe（含 one-hot）
row = {
    "age": age,
    "length_of_stay": length_of_stay,
    "num_previous_admissions": num_adm,
    "medication_compliance_score": compliance,
    "family_support_score": support,
    "post_discharge_followups": followups,
    "both_self_harm": int((recent_self_harm=="Yes") and (selfharm_adm=="Yes")),
    f"gender_{gender}": 1,
    f"diagnosis_{diagnosis}": 1,
    f"has_social_worker_{social_worker}": 1,
    f"has_recent_self_harm_{recent_self_harm}": 1,
    f"self_harm_during_admission_{selfharm_adm}": 1,
}
user = pd.DataFrame([row])

# 與訓練欄位對齊（缺的補 0，順序照 sample）
X_final = pd.DataFrame(columns=sample.columns); X_final.loc[0] = 0
for c in user.columns:
    if c in X_final.columns:
        X_final.at[0, c] = user[c].iloc[0]

# 預測機率
prob = float(model.predict_proba(X_final)[:,1][0])
st.metric("Predicted Dropout Risk (within 3 months)", f"{prob*100:.1f}%")

# 先用機率分級
if prob >= HIGH_CUT:
    level = "high"
elif prob >= YOUDEN_CUT:
    level = "medium"
else:
    level = "low"

# 🔒 臨床安全覆寫：雙自傷一律 High
if (recent_self_harm=="Yes") and (selfharm_adm=="Yes"):
    level = "high"
    st.caption("⚠️ Clinical safety override: recent + in‑admission self-harm → High risk")

# 顯示建議
if level=="high":
    st.error(f"🔴 High Risk (≥{HIGH_CUT:.2f}) — 建議 48–72h 內主動聯繫 + 7天內門診")
elif level=="medium":
    st.warning(f"🟡 Medium Risk (≥{YOUDEN_CUT:.2f}) — 建議 72h 內提醒 + 14天內回診")
else:
    st.success(f"🟢 Low Risk (<{YOUDEN_CUT:.2f}) — 常規 30天提醒")

st.divider()
st.caption("Note: Research prototype with simulated data, calibrated and thresholded via Sensitivity≥0.85 + Decision Curve + Youden’s J. Not for clinical use.")
