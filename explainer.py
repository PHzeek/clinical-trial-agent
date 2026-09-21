"""
explainer.py
SHAP-based explainability layer for the clinical trial scoring function.

What SHAP does here:
  - Takes the scorer's feature matrix (patients x features)
  - Computes how much each feature CONTRIBUTED to each trial's score
  - Positive SHAP value = this feature HELPED the match
  - Negative SHAP value = this feature HURT the match

We use KernelExplainer because:
  - Our scorer is a simple weighted sum (model-agnostic)
  - KernelExplainer works with ANY prediction function
  - It doesn't require a traditional ML model underneath
"""

import shap
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for Streamlit compatibility
import matplotlib.pyplot as plt
import io
import base64

from scorer import build_feature_matrix, WEIGHTS


# ── Prediction function SHAP will explain ────────────────────────────────

def make_predict_fn(feature_names: list) -> callable:
    """
    Creates a prediction function that SHAP can call.
    Takes a numpy matrix X and returns a score array.
    """
    weights = np.array([WEIGHTS[f] for f in feature_names])

    def predict(X: np.ndarray) -> np.ndarray:
        return X @ weights # Simple weighted dot product
    
    return predict


# ── Compute SHAP values for all trials ───────────────────────────────────

def compute_shap_values(patient: dict, trials: list) -> dict:
    """
    Compute SHAP values explaining why each trial got its score.

    Returns:
        {
            "shap_values": np.ndarray(n_trials x n_features),
            "feature_names": list,
            "trial_ids": list,
            "X": np.ndarray,
            "expected_value": float  
        } 
    """
    if not trials:
        return {}
    
    X, feature_names, trial_ids = build_feature_matrix(patient, trials)
    predict_fn = make_predict_fn(feature_names)

    # KernelExplainer: model-agnostic SHAP
    # Background = mean feature values( what "average" looks like)
    background = np.mean(X, axis=0, keepdims=True)
    explainer = shap.KernelExplainer(predict_fn, background)

    # Suppress SHAP progress output for cleaner UI
    shap_values = explainer.shap_values(X, silent=True)

    return {
        "shap_values":   shap_values,
        "feature_names": feature_names,
        "trial_ids":     trial_ids,
        "X":             X,
        "expected_value": explainer.expected_value  
    }


# ── Shared display labels (one dict, used by every chart) ────────────────

READABLE_LABELS = {
    "age_match":           "Age Range",
    "condition_match":     "Condition Relevance",
    "lab_values_ok":       "Lab Values",
    "prior_tx_ok":         "Prior Treatments",
    "ecog_ok":             "ECOG Status",
    "no_exclusion_flags":  "No Exclusions",
}


# ── Resolve a trial's SHAP row by NCT ID (never by list position) ────────

def trial_index_for(shap_result: dict, nct_id: str) -> int | None:
    """
    Return the row of the SHAP matrix that belongs to this NCT ID.

    The SHAP matrix is in API-retrieval order. Claude's ranked list is in a
    different order, so position N in one is NOT position N in the other.
    Always look the row up by ID. Returns None if the trial wasn't scored.
    """
    wanted = (nct_id or "").strip().upper()
    for i, tid in enumerate(shap_result.get("trial_ids", [])):
        if str(tid).strip().upper() == wanted:
            return i
    return None


# ── Generate SHAP waterfall chart for a single trial ─────────────────────

def shap_waterfall_chart(shap_result: dict, trial_index: int, trial_title: str) -> str:
    """
    Generate a SHAP waterfall chart for one trial.
    trial_index must come from trial_index_for() - it is a SHAP matrix row.
    Returns base64-encoded PNG string for display in Streamlit.
    """
    shap_vals     = np.asarray(shap_result["shap_values"][trial_index], dtype=float)
    feature_names = shap_result["feature_names"]

    # Sort features by absolute SHAP value (most impactful first)
    sorted_idx   = np.argsort(np.abs(shap_vals))[::-1]
    sorted_feats = [feature_names[i] for i in sorted_idx]
    sorted_vals  = shap_vals[sorted_idx]
    labels       = [READABLE_LABELS.get(f, f) for f in sorted_feats]

    GREEN, RED, GRAY = "#2ecc71", "#e74c3c", "#7f8c8d"

    def color_for(v):
        if abs(v) < 0.05:          # rounds to 0.0 -> neutral, not "helped"
            return GRAY
        return GREEN if v > 0 else RED

    def text_for(v):
        if abs(v) < 0.05:
            return "0.0"
        return f"+{v:.1f}" if v > 0 else f"\u2212{abs(v):.1f}"   # real minus sign

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, sorted_vals,
                   color=[color_for(v) for v in sorted_vals],
                   edgecolor="white", height=0.6)

    # Value labels. bar_label pads in POINTS (screen units), not data units,
    # so the gap is identical on the positive and negative side and doesn't
    # change when the x-axis scale changes from patient to patient.
    texts = ax.bar_label(bars, labels=[text_for(v) for v in sorted_vals],
                         padding=5, fontsize=9, fontweight="bold")
    for t, v in zip(texts, sorted_vals):
        t.set_color(color_for(v))

    # Reserve room INSIDE the axes for those labels so a long negative bar's
    # label can't run into the y-axis feature names (the "-22.5 / Age Range"
    # overlap), and a long positive bar's label can't clip at the right edge.
    lo, hi = min(sorted_vals.min(), 0.0), max(sorted_vals.max(), 0.0)
    span   = max(hi - lo, 1.0)
    ax.set_xlim(lo - 0.16 * span, hi + 0.16 * span)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (contribution to match score)", fontsize=9)
    ax.set_title(
        f"Why this trial matched\n{trial_title[:55]}{'...' if len(trial_title) > 55 else ''}",
        fontsize=10, fontweight="bold", pad=10
    )
    ax.invert_yaxis()
    plt.tight_layout()

    # Encode to base64 for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Generate SHAP summary bar chart across all trials ────────────────────

def shap_summary_chart(shap_result: dict) -> str:
    """
    Generate a bar chart showing mean |SHAP value| per feature across all trials.
    This shows WHICH features matter most overall for this patient.
    Returns base64-encoded PNG string.
    """
    shap_vals     = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)

    labels = [READABLE_LABELS.get(feature_names[i], feature_names[i]) for i in sorted_idx]
    values = mean_abs_shap[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(labels, values, color="#3498db", edgecolor="white", height=0.5)
    ax.set_xlabel("Mean |SHAP Value| (average impact across all trials)", fontsize=9)
    ax.set_title("Feature Importance - What Drives Matches for This Patient", fontsize=10, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


if __name__ == "__main__":
    from synthetic_patients import generate_patient
    from trials_api import fetch_trials

    patients = generate_patient(1)
    trials = fetch_trials(patients)[:10] # Limit for speed

    if trials:
        print("Computing SHAP values...")
        result = compute_shap_values(patients, trials)
        print(f"SHAP values shape: {result['shap_values'].shape}")
        print(f"Features: {result['feature_names']}")
        print(f"Expected value (baseline score): {result['expected_value']:.2f}")

