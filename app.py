"""
app.py
Streamlit UI for the Clinical Trial Matching Agent.

Pages:
  1. Patient Input  — Enter patient data via form
  2. Trial Results  — View ranked trial recommendations from Claude
  3. SHAP Analysis  — Visual explanation of what drove each recommendation

Run with:
  streamlit run app.py
"""

import streamlit as st
import json
import base64
import time

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Clinical Trial Matcher",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Imports (after page config) ───────────────────────────────────────────
from synthetic_patients import generate_patient
from trials_api import fetch_trials
from agent import run_agent
from scorer import score_all_trials
from explainer import (compute_shap_values, shap_waterfall_chart, shap_summary_chart,
                       trial_index_for, READABLE_LABELS)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
            font-size: 2rem; font-weight: 700; color: #1a3c5e;
            border-bottom: 3px solid #3498db; padding-bottom: 0.5rem;
}
    .section-header {
            font-size: 1.2rem; font-weight: 600; color: #2c3e50;
            margin-top: 1.5rem;
 }
    .match-strong  { background: #d5f5e3; border-left: 4px solid #2ecc71; padding: 0.8rem; border-radius: 4px; margin: 0.5rem 0; }
    .match-moderate{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 0.8rem; border-radius: 4px; margin: 0.5rem 0; }
    .match-weak    { background: #fdebd0; border-left: 4px solid #e74c3c; padding: 0.8rem; border-radius: 4px; margin: 0.5rem 0; }
    .flag-box  { background: #fef9e7; border: 1px solid #f39c12; padding: 0.4rem 0.8rem; border-radius: 4px; font-size: 0.85rem; margin: 0.2rem 0; }
    .info-box  { background: #eaf4fc; border: 1px solid #3498db; padding: 0.6rem 1rem; border-radius: 4px; font-size: 0.9rem; }
    .synthetic-banner { background: #fdecea; border: 1px solid #e74c3c; color: #922b21; padding: 0.5rem 0.9rem; border-radius: 6px; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.8rem; }
    .match-pill { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

SYNTHETIC_BANNER = (
    '<div class="synthetic-banner">🧪 DEMO - SYNTHETIC PATIENT DATA ONLY. '
    'No real patient information is used. Not for clinical decision-making.</div>'
)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Navigation and session state controls
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.markdown("## Clinical Trial Matcher")
    st.markdown("*Powered by Claude + SHAP*")
    st.caption("🧪 Synthetic patient data - demo only")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🧑‍⚕️ Patient Input", "🔬 Trial Results","📊 SHAP Analysis"],
        key="page"
    )

    st.divider()

    # Quick action: generate a random paient for demo
    if st.button("🎲 Generate Random Patient", use_container_width=True):
        st.session_state["patient"] = generate_patient(
            int(time.time()) % 1000
        )
        st.session_state["trials"]         = None
        st.session_state["ranked"]         = None
        st.session_state["scored"]         = None
        st.session_state["shap_result"]    = None
        st.success("Random patient generated!")
        st.rerun()

    if st.button("🔄 Reset Everything", use_container_width=True):
        for key in ["patient", "trials", "ranked", "scored", "shap_result"]:
            st.session_state.pop(key, None)
        st.rerun()

    # Status indicators
    st.divider()
    st.markdown("**Pipeline Status**")
    col1, col2 = st.columns(2)
    col1.metric("Patient", "✅" if "patient" in st.session_state else "⬜")
    col2.metric("Trials",  "✅" if st.session_state.get("trials") else "⬜")
    col1.metric("Ranked",   "✅" if st.session_state.get("ranked") else "⬜")
    col2.metric("SHAP",    "✅" if st.session_state.get("shap_result") else "⬜")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: PATIENT INPUT
# ════════════════════════════════════════════════════════════════════════════

if page == "🧑‍⚕️ Patient Input":

    st.markdown('<div class="main-header">🧑‍⚕️ Patient Data Entry</div>', unsafe_allow_html=True)
    st.markdown(SYNTHETIC_BANNER, unsafe_allow_html=True)
    st.markdown("Fill in the patient's clinical profile below, or use **Generate Random Patient** in the sidebar for a demo.")

    # Pre-fill form if a patient already exists in session state
    p = st.session_state.get("patient", {})
    labs = p.get("lab_values", {})

    with st.form("patient_form"):
        # ── TIER 1: Hard Filters ──────────────────────────────────────────
        st.markdown('<div class="section-header">🔴 Tier 1 - Core Identifiers</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        age = col1.number_input("Age", min_value=1, max_value=110, value=int(p.get("age", 55)))
        sex = col2.selectbox("Sex", ["Male", "Female"], index=0 if p.get("sex", "Male") == "Male" else 1)

        conditions = [
            "Type 2 Diabetes", "Non-Small Cell Lung Cancer", "Breast Cancer",
            "Hypertension", "Chronic Kidney Disease", "Alzheimer's Disease",
            "Rheumatoid Arthritis", "Multiple Sclerosis", "Heart Failure", "Melanoma"
        ]
        condition_default = p.get("diagnosed_condition", "Type 2 Diabetes")
        condition_idx = conditions.index(condition_default) if condition_default in conditions else 0
        condition = col3.selectbox("Diagnosed Condition", conditions, index=condition_idx)

        stages = ["Stage I", "Stage II", "Stage III", "Stage IV", "N/A"]
        stage_default = p.get("disease_stage", "N/A")
        stage_idx = stages.index(stage_default) if stage_default in stages else 4
        disease_stage = st.selectbox("Disease Stage", stages, index=stage_idx)

        # ── TIER 2: Eligibility Validators ───────────────────────────────
        st.markdown('<div class="section-header">🟡 Tier 2 - Clinical Validators</div>', unsafe_allow_html=True)

        st.markdown('**Lab Values**')
        lc1, lc2, lc3, lc4 = st.columns(4)

        egfr       = lc1.number_input("eGFR (ml/min)",       min_value=5,   max_value=120, value=int(labs.get("egfr_ml_min", 60)))
        creatinine = lc2.number_input("Creatinine (mg/dL)",  min_value=0.3, max_value=10.0, value=float(labs.get("creatinine_mg_dl", 1.0)), step=0.1)
        hemoglobin = lc3.number_input("Hemoglobin (g/dL)",   min_value=5.0, max_value=20.0, value=float(labs.get("hemoglobin_g_dl", 13.0)), step=0.1)
        platelets  = lc4.number_input("Platelets (K/µL)",    min_value=20,  max_value=600, value=int(labs.get("platelets_k_ul", 200)))

        lc5, lc6, lc7, lc8 = st.columns(4)
        alt          = lc5.number_input("ALT (U/L)",         min_value=5,    max_value=500, value=int(labs.get("alt_u_l", 30)))
        ast          = lc6.number_input("AST (U/L)",         min_value=5,    max_value=500, value=int(labs.get("ast_u_l", 30)))
        qt_interval  = lc7.number_input("EKG QT (ms)",       min_value=300,  max_value=600, value=int(labs.get("ekg_qt_interval_ms", 420)))
        hba1c        = lc8.number_input("HbA1c (%)",         min_value=4.0,  max_value=15.0, value=float(labs.get("hba1c_pct", 5.5)), step=0.1)

        prior_tx_options = [
            "Metformin", "Insulin", "GLP-1 agonist", "SGLT2 inhibitor",
            "Carboplatin", "Pembrolizumab", "Erlotinib", "Tamoxifen",
            "Herceptin", "Doxorubicin", "Lisinopril", "Methotrexate",
            "Adalimumab", "Interferon beta", "Furosemide", "Nivolumab"    
        ]
        prior_treatments = st. multiselect(
            "Prior Treatments",
            prior_tx_options,
            default=[t for t in p.get("prior_treatments", []) if t in prior_tx_options]  
        )

        comorbidity_options = ["Hypertension", "Type 2 Diabetes", "COPD", "Atrial Fibrillation", "Depression", "None"]
        comorbidities = st.multiselect(
            "Comorbidities",
            comorbidity_options,
            default=[c for c in p.get("comorbidities", ["None"]) if c in comorbidity_options]
        )

        # ── TIER 3: Optimization Factors ─────────────────────────────────
        st.markdown('<div class="section-header">🟢 Tier 3 - Optimization Factors</div>', unsafe_allow_html=True)
        oc1, oc2, oc3 = st.columns(3)

        ecog = oc1.selectbox("ECOG Performance Status", [0, 1, 2],
                            index=p.get("ecog_performance_status", 0),
                            help="0=Fully active, 1=Retricted but ambulatory, 2=Ambulatory but limited self-care")   
        activity = oc2.selectbox("Activity Level", 
                                 ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"],
                                 index=["Sedentary", "Lightly Active", "Moderately Active", "Very Active"].index(
                                     p.get("activity_level", "Moderately Active")))
        smoking = oc3.selectbox("Smoking Status", ["Never", "Former", "Current"],
                                index=["Never", "Former", "Current"].index(p.get("smoking_status", "Never")))
        
        willing_to_travel = st.checkbox("Willing to travel for trial", value=p.get("willing_to_travel",  True))

        submitted = st.form_submit_button("💾 Save Patient Profile", use_container_width=True, type="primary")

    if submitted:
        st.session_state["patient"]= {
            "patient_id": p.get("patient_id", "PT-MANUAL"),
            "age": age,
            "sex": sex,
            "diagnosed_condition": condition,
            "disease_stage": disease_stage,
            "ecog_performance_status": ecog,
            "prior_treatments": prior_treatments,
            "comorbidities": comorbidities,
            "activity_level": activity,
            "smoking_status": smoking,
            "willing_to_travel": willing_to_travel,
            "lab_values": {
                "egfr_ml_min": egfr,
                "creatinine_mg_dl": creatinine,
                "hemoglobin_g_dl": hemoglobin,
                "platelets_k_ul": platelets,
                "alt_u_l": alt,
                "ast_u_l": ast,
                "ekg_qt_interval_ms": qt_interval,
                "hba1c_pct": hba1c,
            }
        }
        st.session_state["trials"]            = None
        st.session_state["ranked"]            = None
        st.session_state["scored"]            = None
        st.session_state["shap_result"]       = None
        st.success("Patient profile saved! Navigate to **Trial Results** to run the agent.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: TRIAL RESULTS
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Trial Results":

    st.markdown('<div class="main-header">🔬 Clinical Trial Recommendations</div>', unsafe_allow_html=True)
    st.markdown(SYNTHETIC_BANNER, unsafe_allow_html=True)

    if "patient" not in st.session_state:
        st.warning("⚠️ No patient loaded. Go to **Patient Input** first.")
        st.stop()

    patient = st.session_state["patient"]

    # Patient summary card
    with st.expander("📋 Active Patient Profile", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient ID", patient.get("patient_id", "N/A"))
        c2.metric("Age / Sex", f"{patient['age']} / {patient['sex']}")
        c3.metric("Condition", patient.get("diagnosed_condition", "N/A"))
        c4.metric("ECOG",      patient.get("ecog_performance_status", "N/A"))

        #Run pipeline button
        if st.button("🚀 Find Matching Trials", type="primary", use_container_width=True):

            with st.status("Running clinical trial matching pipeline...", expanded=True) as status:

                st.write("📡 Fetching live trials from ClinicalTrials.gov...")
                trials = fetch_trials(patient)
                st.write(f"   -> Retrieved {len(trials)} candidate trials")

                if not trials:
                    status.update(label="No trials found. Try a different patient profile.", state="error")
                    st.stop()

                st.write("⚖️ Scoring trials against profile...")
                scored = score_all_trials(patient, trials)
                st.write(f"    -> Scored and sorted {len(scored)} trials")

                st.write("🧠 Claude is reasoning through eligibility and ranking...")
                # run_agent only returns trials that exist in `trials` - every
                # entry is verified against the retrieved list before it gets here.
                ranked = run_agent(patient, trials)
                st.write(f"  -> Claude ranked top {len(ranked)} trials (all verified against the retrieved list)")

                st.write("📊 Computing SHAP explanations...")
                # Explain EVERY candidate Claude saw. Slicing to trials[:10] meant
                # a trial Claude ranked could be missing from the SHAP matrix.
                shap_result = compute_shap_values(patient, trials)
                st.write("   -> SHAP values computed")

                # Commit all four results TOGETHER, only once the whole run has
                # finished. Writing them one by one meant an interrupted run could
                # leave a new trial list sitting next to an old ranking.
                st.session_state.update({
                    "trials": trials, "scored": scored,
                    "ranked": ranked, "shap_result": shap_result,
                })

                status.update(label="Pipeline complete! ✅", state="complete")

        # Display results
        if st.session_state.get("ranked"):
            ranked = st.session_state["ranked"]

            st.markdown(f"### Top {len(ranked)} Matched Trials")
            st.markdown('<div style="background-color:#1a3a5c;padding:12px 16px;border-radius:8px;color:#e8f4fd;font-size:14px;border-left:4px solid #4a9eda;">📊 Ranked by Claude using your clinical hierarchy: Age → Condition → Labs → Prior Treatments → ECOG → Lifestyle</div>', unsafe_allow_html=True)
            st.markdown("")

            for match in ranked:
                score_label = match.get("match_score", "Moderate")
                css_class = {"Strong": "match-strong", "Moderate":"match-moderate", "Weak": "match-weak"}.get(score_label, "match-moderate")
                score_emoji = {"Strong": "🟢", "Moderate": "🟡", "Weak": "🔴"}.get(score_label, "🟡")

                with st.expander(
                    f"{score_emoji} #{match.get('rank', '?')} [{score_label}] {match.get('title', 'N/A')[:70]}",
                    expanded=match.get("rank") ==1
                ):
                    st.markdown(f'<span class="match-pill {css_class}">{score_emoji} {score_label} match</span>', unsafe_allow_html=True)

                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**NCT ID:** `{match.get('nct_id', 'N/A')}`")
                        st.markdown(f"**Why matched:** {match.get('why_matched', 'N/A')}")

                        flags = match.get("eligibility_flags", [])
                        if flags:
                            st.markdown("**⚠️ Eligibility Flags:**")
                            for f in flags:
                                st.markdown(f'<div style="background-color:#fff3cd;padding:8px 12px;border-radius:6px;margin:4px 0;color:#856404;">⚠ {f}</div>', unsafe_allow_html=True)

                        notes = match.get("ambiguity_notes", [])
                        if notes:
                            st.markdown("**~ Ambiguity Notes:**")
                            for n in notes:
                                st.markdown(f'<div style="background-color:#fff3cd;padding:8px 12px;border-radius:6px;margin:4px 0;color:#856404;">~ {n}</div>', unsafe_allow_html=True)

                    with col_b:
                        st.markdown(f"**Recommended Next Step:**")
                        next_step = (match.get("recommended_next_step")
                                     or match.get("recommend_next_step")
                                     or "Confirm full eligibility with the trial's study coordinator before referral.")
                        st.info(next_step)
                        nct_id = match.get("nct_id", "")
                        if nct_id and nct_id != "N/A":
                            st.markdown(f"[🔗 View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{nct_id})")

            st.success("Navigate to **📊 SHAP Analysis** to understand what drove these recommendations.")

        
# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: SHAP ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

elif page == "📊 SHAP Analysis":

    st.markdown('<div class="main-header">📊 SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown(SYNTHETIC_BANNER, unsafe_allow_html=True)
    st.markdown("SHAP shows why each trial received its score - which patient features helped or hurt each match, "
                "**relative to the average candidate trial** for this patient.")

    if not st.session_state.get("shap_result"):
        st.warning("⚠️ No SHAP data yet. Run the pipeline on the **Trial Results** page first.")
        st.stop()

    shap_result = st.session_state["shap_result"]
    trials      = st.session_state.get("trials") or []
    ranked      = st.session_state.get("ranked") or []

    # Self-heal: if any trial Claude ranked is a real candidate but has no SHAP
    # row (e.g. a SHAP result left over from an earlier run that only covered
    # the first 10 trials), recompute SHAP over ALL candidates right now.
    candidate_ids = {str(t.get("nct_id", "")).strip().upper() for t in trials}
    shap_ids      = {str(t).strip().upper() for t in shap_result.get("trial_ids", [])}
    ranked_ids    = {str(r.get("nct_id", "")).strip().upper() for r in ranked}
    if (ranked_ids & candidate_ids) - shap_ids and "patient" in st.session_state:
        with st.spinner("Refreshing SHAP values for all candidate trials..."):
            shap_result = compute_shap_values(st.session_state["patient"], trials)
            st.session_state["shap_result"] = shap_result

    # ── Overall Feature Importance ────────────────────────────────────────
    st.markdown("### 🌐 Overall Feature Importance")
    st.markdown("Which patient features had the biggest impact across **all** candidate trials?")

    summary_b64 = shap_summary_chart(shap_result)
    st.image(
        base64.b64decode(summary_b64),
        caption="Mean |SHAP Value| per feature - larger bars = more influential",
        use_container_width=True
    )

    st.divider()

    # ── Per-Trial Waterfall Charts ─────────────────────────────────────────
    st.markdown("###  🔍 Per-Trial Breakdown")
    st.markdown("Select a trial to see exactly which features drove its match score:")

    if not ranked:
        st.info("No ranked trials available for individual breakdown.")
    else:
        # Map each of Claude's ranked trials to ITS row in the SHAP matrix by NCT ID.
        # Claude's order and the SHAP matrix order are different lists - position
        # N in one is not position N in the other.
        trial_options = {}
        unmatched = []
        for i, r in enumerate(ranked):
            row = trial_index_for(shap_result, r.get("nct_id", ""))
            if row is None:
                unmatched.append(r.get("nct_id", "N/A"))
                continue
            label = f"#{r.get('rank', i + 1)} [{r.get('match_score', 'N/A')}] {r.get('title', 'N/A')[:45]}"
            trial_options[label] = (row, r)

        if unmatched:
            # Only reachable if Claude returned an NCT ID that was never in the
            # candidate list it was given.
            st.warning(f"These ranked trials aren't in the current candidate list, so they have no "
                       f"SHAP row: {', '.join(unmatched)}. The ranking is from an earlier run - "
                       f"click **Find Matching Trials** again.")
        if not trial_options:
            st.warning("None of the ranked trials could be matched to SHAP rows by NCT ID. Re-run the pipeline.")
            st.stop()

        selected_label      = st.selectbox("Select a trial to explain:", list(trial_options.keys()))
        trial_index, chosen = trial_options[selected_label]
        trial_title         = chosen.get("title", "N/A")
        chosen_nct          = chosen.get("nct_id", "N/A")

        # Show the scorer's own view of this exact trial so both layers are
        # visibly talking about the same thing.
        src_trial  = next((t for t in trials if t.get("nct_id") == chosen_nct), {})
        src_scored = next((s for s in (st.session_state.get("scored") or []) if s.get("nct_id") == chosen_nct), {})
        patient    = st.session_state.get("patient", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("NCT ID", chosen_nct)
        m2.metric("Rule-based score", f"{src_scored.get('score', 0):.0f} / 100" if src_scored else "N/A")
        m3.metric("Trial age range vs patient",
                  f"{src_trial.get('min_age') or 'N/A'} - {src_trial.get('max_age') or 'N/A'}",
                  f"patient is {patient.get('age', '?')}", delta_color="off")

        with st.spinner("Generating SHAP waterfall chart..."):
            waterfall_b64 = shap_waterfall_chart(shap_result, trial_index, trial_title)

        st.image(
            base64.b64decode(waterfall_b64),
            caption="Green bars = features that HELPED this match | Red bars = features that HURT this match",
            use_container_width=True
        )

        # ── Interpretation helper ──────────────────────────────────────────
        st.markdown("### 💡 How to read this chart")
        st.markdown("""
        | Color | Meaning |
        |-------|---------|
        | 🟢 Green bar (positive) | This feature **increased** the match score - patient meets this criterion well |
        | 🔴 Red bar (negative)   | This feature **decreased** the match score - patient may not meet this criterion |
        | Bar length              | How much this feature influenced the score - longer = more influential |               
""")

        # ── Raw SHAP values table ─────────────────────────────────────────
        with st.expander("🔢 Raw SHAP Values for This Trial"):
            import pandas as pd
            shap_row = shap_result["shap_values"][trial_index]
            df = pd.DataFrame({
                "Feature":    [READABLE_LABELS.get(f, f) for f in shap_result["feature_names"]],
                "SHAP Value": [round(v, 3) for v in shap_row],
                "Direction":  ["➖ Neutral" if abs(v) < 0.05 else ("✅ Helped" if v > 0 else "❌ Hurt") for v in shap_row]
            }).sort_values("SHAP Value", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
