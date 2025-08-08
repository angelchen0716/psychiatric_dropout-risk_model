# 03_pick_thresholds.py
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.metrics import confusion_matrix, roc_auc_score

# 載入
model = joblib.load("dropout_model.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")["label"].values
y_prob = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
prev = y_test.mean()

def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    ppv  = tp/(tp+fp) if (tp+fp)>0 else 0
    npv  = tn/(tn+fn) if (tn+fn)>0 else 0
    return tp, fp, tn, fn, sens, spec, ppv, npv

def net_benefit(y, p, thr):
    tp, fp, tn, fn, *_ = metrics_at(y, p, thr)
    N = len(y)
    odds = thr/(1-thr)
    return (tp/N) - (fp/N)*odds

rows, thr_list = [], np.linspace(0.01, 0.99, 99)
for t in thr_list:
    tp, fp, tn, fn, sens, spec, ppv, npv = metrics_at(y_test, y_prob, t)
    youden = sens + spec - 1
    nb = net_benefit(y_test, y_prob, t)
    rows.append({"threshold":t,"tp":tp,"fp":fp,"tn":tn,"fn":fn,
                 "sensitivity":sens,"specificity":spec,"ppv":ppv,"npv":npv,
                 "youdenJ":youden,"net_benefit":nb})
df = pd.DataFrame(rows)
df.to_csv("threshold_table.csv", index=False)

# High cut：Sensitivity >= 0.85 的最小門檻（在接近區間取 NB 最大）
cand = df[df["sensitivity"]>=0.85]
if not cand.empty:
    tmin = cand["threshold"].min()
    high_cut = cand[cand["threshold"]<=tmin+0.05].sort_values(
        ["net_benefit","threshold"], ascending=[False,True]
    ).iloc[0]["threshold"]
else:
    high_cut = df.sort_values("youdenJ", ascending=False).iloc[0]["threshold"]

# Youden cut：最大 Youden’s J
youden_cut = df.sort_values("youdenJ", ascending=False).iloc[0]["threshold"]

# Decision curve
plt.figure(figsize=(6,5))
plt.plot(df["threshold"], df["net_benefit"], lw=2, label="Model")
treat_all = prev - (1-prev)*(df["threshold"]/(1-df["threshold"]))
plt.plot(df["threshold"], treat_all, "--", label="Treat All")
plt.plot(df["threshold"], 0*df["threshold"], "--", label="Treat None")
plt.axvline(high_cut, color="red", ls=":", label=f"High cut={high_cut:.2f}")
plt.axvline(youden_cut, color="orange", ls=":", label=f"Youden cut={youden_cut:.2f}")
plt.xlabel("Threshold probability"); plt.ylabel("Net Benefit")
plt.title(f"Decision Curve (AUC={auc:.2f}, Prev={prev:.2f})")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("decision_curve.png", dpi=150); plt.show()

with open("thresholds.json","w") as f:
    json.dump({
        "high_cut": round(float(high_cut),3),
        "youden_cut": round(float(youden_cut),3),
        "prevalence": round(float(prev),3),
        "auc": round(float(auc),3)
    }, f, indent=2)

print("✅ thresholds.json, threshold_table.csv, decision_curve.png 已產生")
print(f"建議 High cut={high_cut:.3f}（Sens≥0.85 原則）、Youden cut={youden_cut:.3f}")
