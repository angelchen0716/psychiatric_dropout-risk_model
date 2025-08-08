# 02_evaluate_model.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

model = joblib.load("dropout_model.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")["label"].values

# 機率
y_prob = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
print(f"AUC = {auc:.3f}")

# --- ROC ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150); plt.show()

# --- Confusion Matrix（預設 0.5；報告可再放 high_cut 的版本）---
thr = 0.5
y_pred = (y_prob >= thr).astype(int)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Dropout","Dropout"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (threshold={thr:.2f})")
plt.tight_layout(); plt.savefig("confusion_matrix_0p50.png", dpi=150); plt.show()

# --- 校準曲線 ---
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, "o-", label="Model")
plt.plot([0,1],[0,1],'k--', label="Perfectly calibrated")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration Curve")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=150); plt.show()

print("Saved: roc_curve.png, confusion_matrix_0p50.png, calibration_curve.png")
