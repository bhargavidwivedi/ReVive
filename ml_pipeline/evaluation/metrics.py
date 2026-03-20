import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. LOAD MODELS & DATA ─────────────────────────────────────────────────────

def load_artifacts(model_name="XGBoost"):
    model  = joblib.load(f"ml_pipeline/models/saved/{model_name}.pkl")
    scaler = joblib.load("ml_pipeline/models/saved/scaler.pkl")
    df     = pd.read_csv("data/processed_features.csv")

    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["readmitted_30d"])
    y = df["readmitted_30d"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    feature_names = list(X.columns)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"Loaded: {model_name} | Test rows: {X_test.shape[0]:,}")
    return model, scaler, X_test, y_test, feature_names


# ── 2. ROC CURVE ──────────────────────────────────────────────────────────────

def plot_roc(model, X_test, y_test, model_name="XGBoost"):
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    plt.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — 30-Day Readmission Prediction")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
    plt.show()
    print(f"ROC-AUC: {auc:.4f}")
    return auc


# ── 3. PRECISION-RECALL CURVE ─────────────────────────────────────────────────

def plot_pr_curve(model, X_test, y_test, model_name="XGBoost"):
    y_prob = model.predict_proba(X_test)[:, 1]
    ap     = average_precision_score(y_test, y_prob)
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, color="darkorange", lw=2,
             label=f"{model_name} (AP = {ap:.4f})")
    plt.axhline(y_test.mean(), color="k", linestyle="--",
                label=f"Baseline (prevalence = {y_test.mean():.2%})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — 30-Day Readmission")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pr_curve.png", dpi=150)
    plt.show()
    print(f"Average Precision: {ap:.4f}")
    return prec, rec, thresholds


# ── 4. OPTIMAL THRESHOLD ──────────────────────────────────────────────────────

def find_optimal_threshold(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx  = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    print(f"\n── Optimal Threshold ─────────────────────────────────────")
    print(f"  Threshold : {best_thresh:.3f}")
    print(f"  F1        : {f1_scores[best_idx]:.4f}")
    print(f"  Precision : {prec[best_idx]:.4f}")
    print(f"  Recall    : {rec[best_idx]:.4f}")
    return best_thresh


# ── 5. CONFUSION MATRIX ───────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, threshold=0.5, model_name="XGBoost"):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Readmit", "Readmit"],
                yticklabels=["No Readmit", "Readmit"],
                annot_kws={"size": 14})
    plt.title(f"Confusion Matrix — {model_name} (threshold={threshold:.2f})")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.show()

    print(f"\n── Classification Report (threshold={threshold:.2f}) ─────")
    print(classification_report(y_test, y_pred,
                                target_names=["No Readmit", "Readmit"]))

    tn, fp, fn, tp = cm.ravel()
    print(f"  True Positives  (caught readmissions) : {tp:,}")
    print(f"  False Negatives (missed readmissions) : {fn:,}")
    print(f"  False Positives (unnecessary alerts)  : {fp:,}")
    print(f"  True Negatives  (correctly cleared)   : {tn:,}")


# ── 6. SHAP FEATURE IMPORTANCE ────────────────────────────────────────────────

def shap_analysis(model, X_test, feature_names, sample_n=1000, model_name="XGBoost"):
    print(f"\n── SHAP Analysis (sample={sample_n}) ─────────────────────")
    X_sample = pd.DataFrame(X_test, columns=feature_names).sample(
        min(sample_n, len(X_test)), random_state=42
    )
    X_sample = X_sample.apply(pd.to_numeric, errors="coerce").fillna(0)
    

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    sv = shap_values.values if hasattr(shap_values, "values") else shap_values

    # --- Global bar chart (top 20 features) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      plot_type="bar", max_display=20, show=False)
    plt.title("Top 20 Features — Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_importance.png", dpi=150)
    plt.show()

    # --- Beeswarm (direction of impact) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      max_display=20, show=False)
    plt.title("SHAP Beeswarm — Feature Impact Direction")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png", dpi=150)
    plt.show()

    # Print top 10 features by importance
    mean_shap = np.abs(sv).mean(axis=0)
    top_features = pd.DataFrame({
        "feature"    : feature_names,
        "mean_shap"  : mean_shap
    }).sort_values("mean_shap", ascending=False).head(10)

    print("\n  Top 10 features driving readmission risk:")
    print(top_features.to_string(index=False))
    return top_features


# ── 7. FULL REPORT ────────────────────────────────────────────────────────────

def full_evaluation_report(model_name="XGBoost"):
    print("=" * 60)
    print(f"  FULL EVALUATION REPORT — {model_name}")
    print("=" * 60)

    model, scaler, X_test, y_test, feature_names = load_artifacts(model_name)
    X_in = scaler.transform(X_test) if model_name == "LogisticRegression" else X_test.values

    print("\n── 1. ROC Curve ──────────────────────────────────────────")
    plot_roc(model, X_in, y_test, model_name)

    print("\n── 2. Precision-Recall Curve ─────────────────────────────")
    plot_pr_curve(model, X_in, y_test, model_name)

    print("\n── 3. Optimal Threshold ──────────────────────────────────")
    best_thresh = find_optimal_threshold(model, X_in, y_test)

    print("\n── 4. Confusion Matrix ───────────────────────────────────")
    plot_confusion_matrix(model, X_in, y_test, best_thresh, model_name)

    print("\n── 5. SHAP Analysis ──────────────────────────────────────")
    top_features = shap_analysis(model, X_in, feature_names)

    print(f"\n✅ All charts saved to '{OUTPUT_DIR}/' folder")
    return top_features


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    top_features = full_evaluation_report("LightGBM")