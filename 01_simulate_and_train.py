# 01_simulate_and_train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
import joblib
import json

np.random.seed(42)
N = 3000

# --- 1) 模擬原始資料（含 6 類診斷 & 交互項 both_self_harm） ---
raw = pd.DataFrame({
    "age": np.random.randint(18, 80, N),
    "gender": np.random.choice(["Male", "Female"], N),
    "diagnosis": np.random.choice(
        ["Schizophrenia","Bipolar","Depression",
         "Personality Disorder","Substance Use Disorder","Dementia"], N),
    "length_of_stay": np.random.randint(3, 60, N),
    "num_previous_admissions": np.random.randint(0, 10, N),
    "has_social_worker": np.random.choice(["Yes","No"], N, p=[0.6, 0.4]),
    "medication_compliance_score": np.round(np.random.uniform(0,10,N),1),
    "has_recent_self_harm": np.random.choice(["Yes","No"], N, p=[0.3,0.7]),
    "self_harm_during_admission": np.random.choice(["Yes","No"], N, p=[0.1,0.9]),
    "family_support_score": np.round(np.random.uniform(0,10,N),1),
    "post_discharge_followups": np.random.randint(0, 6, N)
})
# 交互項：雙自傷
raw["both_self_harm"] = (
    (raw["has_recent_self_harm"]=="Yes") &
    (raw["self_harm_during_admission"]=="Yes")
).astype(int)

# --- 2) 生成真值: 文獻導向 + 噪聲 + 交互 ---
# 權重（PoC 用，之後交由模型學習）
w = (
    (raw["diagnosis"]=="Schizophrenia")*1.2 +
    (raw["diagnosis"]=="Personality Disorder")*1.1 +
    (raw["diagnosis"]=="Substance Use Disorder")*1.0 +
    (raw["diagnosis"]=="Dementia")*0.5 +
    (raw["diagnosis"]=="Bipolar")*0.6 +
    (raw["diagnosis"]=="Depression")*0.4 +
    (raw["has_social_worker"]=="No")*1.0 +
    (raw["has_recent_self_harm"]=="Yes")*1.4 +
    (raw["self_harm_during_admission"]=="Yes")*1.1 +
    (raw["both_self_harm"]==1)*1.1 +                # 交互加權
    (raw["medication_compliance_score"]<4)*0.8 +
    (raw["family_support_score"]<5)*0.7 +
    (raw["post_discharge_followups"]<2)*1.0 +
    (raw["gender"]=="Male")*0.3 +
    (raw["age"]>60)*0.4
)

# 加一些非線性/噪聲，避免模型「反解規則」
w = w + 0.15*np.sin(raw["age"]/8) + 0.1*np.log1p(raw["num_previous_admissions"]) + np.random.normal(0,0.15,N)

from scipy.special import expit as sigmoid
p = sigmoid((w - 3.7)*1.5)  # shift+scale 控制整體盛行率
y = np.random.binomial(1, p, N)
raw["dropout_within_3_months"] = y

# --- 3) One-hot 編碼 ---
X = pd.get_dummies(raw.drop(columns="dropout_within_3_months"))
y = raw["dropout_within_3_months"].values

# --- 4) Train / Test split（80/20 + 分層）---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 5) 訓練 XGB（基礎） + 機率校準（isotonic）---
base = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42,
    reg_lambda=1.0, n_jobs=-1
)
# 用 5 折做 isotonic calibration（只在訓練集）
calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)

# --- 6) 輸出模型與測試集、樣板 ---
joblib.dump(calibrated, "dropout_model.pkl")
X_test.to_csv("X_test.csv", index=False)
pd.DataFrame({"label": y_test}).to_csv("y_test.csv", index=False)

# 給前端對齊欄位的樣板（5 行；全 0）
sample = pd.DataFrame(columns=X.columns)
sample.loc[0:4] = 0
sample.to_csv("sample_input.csv", index=False)

# 顯示摘要
print("✅ Done.")
print(f"Prevalence (train): {y_train.mean():.3f} | (test): {y_test.mean():.3f}")
print(f"Saved: dropout_model.pkl, X_test.csv, y_test.csv, sample_input.csv")
