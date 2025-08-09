# app.py — Psychiatric Dropout Risk (2 pages: Predictor / Model Evaluation)
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")

# ================== Helpers ==================
@st.cache_resource(show_spinner=False)
def load_model(path="dropout_model.pkl"):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_sample(path="sample_input.csv"):
    df = pd.read_csv(path)
    return df.fillna(0)

def load_thresholds(path="thresholds.json"):
    try:
        with open(path, "r") as f:
            js = json.load(f)
        return float(js["high_cut"]), float(js["youden_cut"]), js
    except Exception:
        st.warning("`thresholds.json` not found — fallback to defaults (High=0.30, Mid=0.20). "
                   "Run 03_pick_thresholds.py to generate proper cuts.")
        return 0.30, 0.20, {"prevalence": None, "auc": None}

@st.cache_resource(show_spinner=False)
def load_test_data():
    try:
        X_test = pd.read_csv("X_test.csv")
        y_test = pd.read_csv("y_test.csv")["label"].values
        return X_test, y_test
    except Exception as e:
        return None, None

# 嘗試從 CalibratedClassifierCV 取出樹模型；或本身就是樹模型
def get_tree_model_for_shap(model):
    try:
        # sklearn >= 1.1 CalibratedClassifierCV
        return model.calibrated_classifiers_[0].base_estimator
    except Exception:
        try:
            from xgboost import XGBClassifier
            if isinstance(model, XGBClassifier):
                return model
        except Exception:
            pass
    return None

@st.cache_resource(show_spinner=False)
def get_tree_explainer(tree_model, background_df):
    import shap
    expl = shap.TreeExplainer(tree_model)
    bg = background_df.copy()
    if len(bg) > 200:      # 限制背景樣本，避免雲端爆記憶體
        bg = bg.sample(200, random_state=42)
    _ = expl(bg)           # warm up
    return expl

# ================== Load essentials ==================
model = load_model()
sample = load_sample()
HIGH_CUT, YOUDEN_CUT, THR_META = load_thresholds()
X_test, y_test = load_test_data()

# ================== UI Layout ==================
st.title("🧠 Psychiatric Dropout Risk")
page = st.sidebar.radio("Pages", ["Predictor", "Model Evaluation"])

# =====================================================================================
# Page 1 — Predictor
# =====================================================================================
if page == "Predictor":
    with st.sidebar:
        st.header("Patient Info")
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        diagnosis = st.selectbox("Diagnosis", [
            "Schizophrenia","Bipolar","Depression",
            "Personality Disorder","Substance Use Disorder","Dementia"
        ])
        length_of_stay = st.slider("Length of Stay (days)", 1, 90, 10)
        num_adm = st.slider("# Previous Admissions", 0, 15, 1)
        social_worker = st.radio("Has Social Worker", ["Yes", "No"])
        compliance = st.slider("Medication Compliance Score", 0.0, 10.0, 5.0, 0.1)
        recent_self_harm = st.radio("Recent Self-harm", ["Yes", "No"])
        selfharm_adm = st.radio("Self-harm During Admission", ["Yes", "No"])
        support = st.slider("Family Support Score", 0.0, 10.0, 5.0, 0.1)
        followups = st.slider("Post-discharge Followups", 0, 10, 2)

        st.markdown("---")
        st.subheader("Safety / Threshold Settings")
        enable_override = st.checkbox(
            "Enable clinical safety override (double self-harm → High)", value=True
        )
        min_override_prob = st.slider(
            "Min prob required to trigger override", 0.00, 0.50, 0.15, 0.01
        )
        show_debug = st.checkbox("Show debug info", value=False)

    # 單筆輸入 → one-hot 對齊
    row = {
        "age": age,
        "length_of_stay": length_of_stay,
        "num_previous_admissions": num_adm,
        "medication_compliance_score": compliance,
        "family_support_score": support,
        "post_discharge_followups": followups,
        "both_self_harm": int((recent_self_harm == "Yes") and (selfharm_adm == "Yes")),
        f"gender_{gender}": 1,
        f"diagnosis_{diagnosis}": 1,
        f"has_social_worker_{social_worker}": 1,
        f"has_recent_self_harm_{recent_self_harm}": 1,
        f"self_harm_during_admission_{selfharm_adm}": 1,
    }
    user = pd.DataFrame([row])

    X_final = pd.DataFrame(columns=sample.columns)
    X_final.loc[0] = 0
    for c in user.columns:
        if c in X_final.columns:
            X_final.at[0, c] = user[c].iloc[0]
    X_final = X_final.fillna(0)

    # 預測機率（校準後機率）
    try:
        prob = float(model.predict_proba(X_final)[:, 1][0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    with c1:
        st.metric("Predicted Dropout Risk (within 3 months)", f"{prob*100:.1f}%")

    # 分級 + 覆寫
    reason = []
    if prob >= HIGH_CUT:
        level = "high"; reason.append(f"prob ≥ high_cut ({prob:.2f} ≥ {HIGH_CUT:.2f})")
    elif prob >= YOUDEN_CUT:
        level = "medium"; reason.append(f"{YOUDEN_CUT:.2f} ≤ prob < {HIGH_CUT:.2f}")
    else:
        level = "low"; reason.append(f"prob < youden_cut ({prob:.2f} < {YOUDEN_CUT:.2f})")

    recent_flag = (recent_self_harm == "Yes")
    adm_flag = (selfharm_adm == "Yes")
    override_fired = False
    if enable_override and recent_flag and adm_flag and prob >= min_override_prob:
        level = "high"
        override_fired = True
        reason.append(f"override: double self-harm AND prob ≥ {min_override_prob:.2f}")

    with c2:
        if level == "high":
            st.error(f"🔴 High Risk (≥ {HIGH_CUT:.2f}) — 建議 48–72h 內主動聯繫 + 7 天內門診")
        elif level == "medium":
            st.warning(f"🟡 Medium Risk (≥ {YOUDEN_CUT:.2f}) — 建議 72h 內提醒 + 14 天內回診")
        else:
            st.success(f"🟢 Low Risk (< {YOUDEN_CUT:.2f}) — 常規 30 天提醒")

    st.caption("**Why this level:** " + " ; ".join(reason))
    if override_fired:
        st.caption("⚠️ Clinical safety override applied.")
    if show_debug:
        meta_txt = []
        if THR_META.get("auc") is not None:
            meta_txt.append(f"AUC(test)={THR_META['auc']:.2f}")
        if THR_META.get("prevalence") is not None:
            meta_txt.append(f"Prev(test)={THR_META['prevalence']:.2f}")
        st.write(
            f"**Debug:** prob={prob:.3f} | high_cut={HIGH_CUT:.3f} | youden_cut={YOUDEN_CUT:.3f} "
            + ("| " + " | ".join(meta_txt) if meta_txt else "")
        )

    st.divider()
    st.subheader("SHAP Explanation (feature contributions)")

    # 取樹模型 → SHAP 單例
    tree_model = get_tree_model_for_shap(model)
    if tree_model is None:
        st.info("SHAP not available for this estimator wrapping. (No inner tree model found)")
    else:
        import shap
        # 背景資料用 sample（模型訓練欄位），避免 model input mismatch
        explainer = get_tree_explainer(tree_model, sample.fillna(0))
        shap_values = explainer(X_final)

        st.markdown("**Waterfall plot (single case)** — shows how each feature pushes the score up/down.")
        fig1 = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        st.pyplot(fig1, clear_figure=True)

        st.markdown("**Top feature contributions (absolute SHAP)**")
        fig2 = plt.figure(figsize=(7, 4))
        shap.plots.bar(shap_values[0], max_display=12, show=False)
        st.pyplot(fig2, clear_figure=True)

        st.caption(
            "Note: SHAP explains the **pre‑calibration tree model**. "
            "Risk grading uses the **calibrated probability** with Sens≥0.85 high‑cut and Youden mid‑cut."
        )

    st.divider()
    st.caption(
        "Prototype for research/education only. "
        "Model trained on literature‑guided simulated data, with isotonic calibration. "
        "For clinical deployment, retrain on local EHR with IRB and external validation."
    )

# =====================================================================================
# Page 2 — Model Evaluation
# =====================================================================================
else:
    st.subheader("Model Evaluation")
    if X_test is None or y_test is None:
        st.warning("Missing `X_test.csv` or `y_test.csv`. Please run `01_simulate_and_train.py` first.")
        st.stop()

    # 1) AUC / ROC Curve
    from sklearn.metrics import roc_curve, roc_auc_score
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("AUC (test)", f"{auc:.2f}")
    with c2:
        st.metric("Prevalence (test)", f"{y_test.mean():.2f}")

    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # 2) Confusion Matrix（顯示 0.50 / youden_cut / high_cut 三種）
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    def cm_plot(thr, title):
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig = plt.figure(figsize=(5.2, 4.6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Dropout", "Dropout"])
        disp.plot(cmap=plt.cm.Blues, values_format="d", ax=plt.gca(), colorbar=False)
        plt.title(title); plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
        st.caption(f"Sensitivity={sens:.2f} | Specificity={spec:.2f} | PPV={ppv:.2f} | NPV={npv:.2f}")

    st.markdown("### Confusion Matrices at key thresholds")
    cm_plot(0.50, "Confusion Matrix @ 0.50")
    cm_plot(YOUDEN_CUT, f"Confusion Matrix @ Youden cut = {YOUDEN_CUT:.2f}")
    cm_plot(HIGH_CUT, f"Confusion Matrix @ High cut = {HIGH_CUT:.2f}")

    # 3) Calibration Curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    fig = plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, "o-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration Curve"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # 4) SHAP Summary (beeswarm) — 全體測試集
    st.markdown("### SHAP Summary (test set)")
    tree_model = get_tree_model_for_shap(model)
    if tree_model is None:
        st.info("SHAP summary not available for this estimator wrapping.")
    else:
        import shap
        # 只取前 800 筆，避免雲端記憶體不足
        X_for_shap = X_test.copy()
        if len(X_for_shap) > 800:
            X_for_shap = X_for_shap.sample(800, random_state=42)

        explainer = get_tree_explainer(tree_model, X_for_shap)
        shap_values = explainer(X_for_shap)

        fig = plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_for_shap, show=False, max_display=20)  # beeswarm
        st.pyplot(fig, clear_figure=True)

    st.divider()
    st.caption(
        "Evaluation based on held‑out test set created during simulation. "
        "For real deployment, replace with hospital EHR data (IRB) and re‑run end‑to‑end."
    )

