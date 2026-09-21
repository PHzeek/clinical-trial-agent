"""
scorer.py
Transparent, rule-based scoring function that quantifies how well a patient
matches a clinical trial. SHAP will explain this scorer's feature contributions.

Why a separate scorer?
  - Claude (LLM) reasons qualitatively. SHAP needs a numeric function.
  - This scorer encodes the clinical hierarchy YOU designed:
      Tier 1 → Hard filters (Age, Sex, Condition)   weight: eliminates
      Tier 2 → Eligibility validators (Labs, Priors) weight: high
      Tier 3 → Optimization factors (Lifestyle)      weight: low
  - Every number here is explainable to a clinician.
"""

import re
import numpy as np


# ── Feature weights (must sum to 100) ─────────────────────────────────────
WEIGHTS = {
    # Tier 1 - Hard Filters
    "age_match":           25,   # Does patient age fall within trial? range?
    "condition_match":     25,   # Does condtionmatch trial focus?
    # Tier 2 - Eligibility Validators
    "lab_values_ok":       20,   # Are key labs within acceptable ranges?
    "prior_tx_ok":      15,   # No exclusionary prior treatments?
    # Tier 3 - Optimization/Ranking
    "ecog_ok":              8,    # ECOG performance status acceptable?
    "no_exclusion_flags":   7,    # No comorbidity-based exclusios?
}

assert sum(WEIGHTS.values()) == 100, "Weights must sum to 100"


# ── Helper: Parse age strings from trial eligibility ──────────────────────

def parse_age_years(age_str: str) -> int | None:
    """Extract numeric age from strings like '18 Years', '65 Years', 'N/A'."""
    if not age_str or age_str.strip().upper() in ("N/A", "NONE", ""):
        return None
    match = re.search(r"(\d+)", age_str)
    return int(match.group(1)) if match else None


# ── Feature extraction: convert patient + trial into a numeric feature dict ──

def extract_features(patient: dict, trial: dict) -> dict:
    """
    Extract numeric features (0.0 to 1.0) for each scoring dimension.
    These are the values SHAP will explain
    """
    features = {}

    # ── FEATURE 1: Age Match (0 or 1) ──────────────────────────────────────
    patient_age = patient.get("age", 0)
    min_age = parse_age_years(trial.get("min_age", ""))
    max_age = parse_age_years(trial.get("max_age", ""))

    if  min_age is None and max_age is None:
        features['age_match'] = 1.0 # No restriction -> assume match
    elif min_age is not None and max_age is not None:
        features["age_match"] = 1.0 if min_age <= patient_age <= max_age else 0.0
    elif min_age is not None:
        features["age_match"] = 1.0 if patient_age >= min_age else 0.0
    else:
        features["age_match"] = 1.0 if patient_age <= max_age else 0.0
    
    # ── FEATURE 2: Condition Match (0 or 1) ────────────────────────────────
    patient_condition = patient.get("diagnosed_condition", "").lower()
    trial_conditions = " ".join(trial.get("condition", [])).lower()\
                        if isinstance(trial.get("condition"), list)\
                        else str(trial.get("condition", "")).lower()
    trial_title      = trial.get("title", "").lower()
    trial_summary    = trial.get("summary", "").lower()

    # Simple keyword overlap - agent will do deeper reasoning
    condition_words = set(patient_condition.replace(",", "").split())
    search_text     = trial_conditions + " " + trial_title + " " + trial_summary
    matched_words   = sum(1 for w in condition_words if w in search_text)

    features["condition_match"] = min(1.0, matched_words / max(len(condition_words), 1))

    # ── FEATURE 3: Lab Values OK (0 to 1) ──────────────────────────────────
    labs = patient.get("lab_values", {})
    eligibility_text = trial.get("eligibility_criteria", "").lower()

    lab_checks = []

    # eGFR - commonly required for renal safety
    egfr = labs.get("egfr_ml_min", 60)
    if "egfr" in eligibility_text or "renal function" in "renal function" in eligibility_text:
        lab_checks.append(1.0 if egfr >= 30 else 0.0)

    # Creatinine - common exclusion criterion
    creatinine = labs.get("creatinine_mg_dl", 1.0)
    if "creatinine" in eligibility_text:
        lab_checks.append(1.0 if creatinine <= 1.5 else 0.0)

    # Hemoglobin - commonly required for oncology trials
    hgb = labs.get("hemoglobin_g_dl", 12.0)
    if "hemoglobin" in eligibility_text or "anemia" in eligibility_text:
        lab_checks.append(1.0 if hgb >= 9.0 else 0.0)

    # Platelets
    plt = labs.get("platelets_k_ul", 200)
    if "platelet" in eligibility_text:
        lab_checks.append(1.0 if plt >= 100 else 0.0)

    # Liver enzymes (ALT/AST)
    alt = labs.get("alt_u_l", 30)
    ast = labs.get("ast_u_l", 30)
    if "liver" in eligibility_text or "alt" in eligibility_text or "ast" in eligibility_text:
        lab_checks.append(1.0 if alt <= 80 and ast <= 80 else 0.0)

    # EKG QT interval - cardiac safety
    qt = labs.get("ekg_qt_interval_ms", 420)
    if "qt" in eligibility_text or "ekg" in eligibility_text or "cardiac" in eligibility_text:
        lab_checks.append(1.0 if qt <= 450 else 0.0)

    features["lab_values_ok"] = np.mean(lab_checks) if lab_checks else 1.0

    # ── FEATURE 4: Prior Treatments OK (0 or 1) ────────────────────────────
    prior_txs = [tx.lower() for tx in patient.get("prior_treatments", [])]
    exclusion_score = 1.0
    for tx in prior_txs:
        if tx in eligibility_text and "exclusion" in eligibility_text:
            exclusion_score = 0.0
            break
    features['prior_tx_ok'] = exclusion_score

     # ── FEATURE 5: ECOG Performance Status OK (0 or 1) ─────────────────────
    ecog = patient.get("ecog_performance_status", 0)
    if "ecog" in eligibility_text:
        # Most trials accept ECOG 0-1; some accept 0-2
        if "ecog 0-2" in eligibility_text or "ecog ≤ 2" in eligibility_text:
            features["ecog_ok"] = 1.0 if ecog <= 2 else 0.0
        else:
            features["ecog_ok"] = 1.0 if ecog <= 1 else 0.0
    else:
        features["ecog_ok"] = 1.0 # No ECOG restrictions

    # ── FEATURE 6: No Exclusion Flags (0 or 1) ─────────────────────────────
    comorbidities = [c.lower() for c in patient.get("comorbidities", [])]
    exclusion_hit = any(c in eligibility_text for c in comorbidities if c != "none")
    features["no_exclusion_flags"] = 0.0 if exclusion_hit else 1.0

    return features


# ── Score a single patient-trial pair ────────────────────────────────────
def score_trial(patient: dict, trial: dict) -> dict:
    """
    Compute a weighted match score for a patient-trial pair.
    Returns score (0-100) and the feature vector.
    """

    features = extract_features(patient, trial)

    score = sum(
        features[feat] * weight
        for feat, weight in WEIGHTS.items()
    )

    return {
        "nct_id": trial.get("nct_id", "N/A"),
        "title": trial.get("title", "N/A"),
        "score": round(score, 2),
        "features": features
    }


# ── Score all trials for a patient ───────────────────────────────────────

def score_all_trials(patient: dict, trials: list) -> list:
    """
    Score and rank all trials for a patient.
    Returns list sorted by score decending.
    """

    scored = [score_trial(patient, trial) for trial in trials]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

# ── Build feature matrix for SHAP ────────────────────────────────────────
def build_feature_matrix(patient: dict, trials: list) -> tuple:
    """
    Build the numpy feature matrix X that SHAP needs.
    Returns (X, feature_names, trial_ids)
    """

    feature_names = list(WEIGHTS.keys())
    rows = []
    trial_ids = []

    for trial in trials:
        features = extract_features(patient, trial)
        rows.append([features[f] for f in feature_names])
        trial_ids.append(trial.get("nct_id", "N/A"))

    X = np.array(rows)
    return X, feature_names, trial_ids


if __name__=="__main__":
    import json
    from synthetic_patients import generate_patient
    from trials_api import fetch_trials

    patient = generate_patient(1)
    trials = fetch_trials(patient)

    if trials:
        scored = score_all_trials(patient, trials)
        print("\n=== TOP 5 SCORED TRIALS ===")
        for s in scored[:5]:
            print(f"  [{s['score']:5.1f}] {s['title'][:60]}")
            print(f"         Features: {s['features']}")
