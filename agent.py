"""
agent.py
The Claude-powered agentic reasoning layer.

This is where the intelligence lives:
- Reasons about which trials match the patient using our clinical hierarchy
- Handles ambiguous eligibility language
- Resolves ties between equally ranked trials
- Returns top 5-10 ranked trials with explanations
"""

import anthropic
import json
import re

client = anthropic.Anthropic() # Reads ANTHROPIC_API_KEY from environment

# ── System prompt: defines the agent's reasoning behavior ──────────────────
SYSTEM_PROMPT = """
You are a clinical trial matching agent. Your role is to evaluate whether a patient
is eligible for clinical trials and rank the best matches. 

You reason through eligibility using this strict hierarchy:

TIER 1 - HARD FILTERS (eliminate immediately if not met):
    1. Age: Patient must fall within the trial's min/max age range 
    2. Sex: Patient's sex must match trial requirements (or trial must accept ALL)
    3. Diagnosed condition: Must match or be directly relevant to the trial's focus

TIER 2 - ELIGIBILITY VALIDATORS (must pass to remain a candidate):
    4. Lab values: Check relevant labs (eGFR, liver enzymes, CBC, EKG)against any
        specific thresholds mentioned in the eligibility criteria
    5. Prior treatments: Identify any exclusionary prior treatments
    6. Comorbidities / contraindications: Flag any listed exclusions

TIER 3 - RANKING FACTORS (used to rank among qualified candidates):
    7. Disease stage alignment
    8. ECOG performance status
    9. Location / willingness to travel
   10. Activity level, lifestyle factors

HANDLING AMBIGUOUS LANGUAGE:
        - If eligibility criteria uses clinical language that doesn't map directly to 
          the patient's record, reason through the most likely clinical interpretation
          and note your uncertainty explicitly.
        - Example: if a trial says "adequate renal function" but doesn't specify a
        threshold, use standard clinical convention (eGFR ≥ 30 or creatinine ≤ 1.5x ULN)
        and flag this assumption.

RESOLVING TIES:
        - When two trials are equally matched on Tiers 1-2, use Tier 3 factors to differentiate.
        - If still tied, prefer trials in earlier phases (more cutting-edge) for younger patients,
          and later phases (more established safety) for older or frailer patients.

OUTPUT FORMAT:
    Return a JSON array of the top 5-10 matching trials, ranked best to worst.
    Each entry must include:
    {
    "rank": 1,
    "candidate_id": "T07",
    "nct_id": "NCT...",
    "title": "...",
    "match_score": "Strong" | "Moderate" | "Weak",
    "why_matched": "Plain language explanation of why this trial fits",
    "eligibility_flags": ["Any concerns or borderline criteria"],
    "ambiguity_notes": ["Any language that required interpretation"],
    "recommended_next_step": "What the clinician should verify before enrolling (REQUIRED - never empty or N/A)"
    }

    Every candidate has a short "candidate_id" (T01, T02, ...). Copy the candidate_id, nct_id
    and title EXACTLY from the candidate you are describing. Only rank trials from the
    provided list - never add a trial from your own knowledge and never alter an NCT ID.
    All candidates have already passed the structured age and sex limits; if the free-text
    criteria state a narrower age range, flag it in eligibility_flags.

    Return ONLY the JSON array. No preamble or extra text.
"""


# ── Grounding: every ranked trial must be one we actually sent ────────────

def tag_candidates(trials: list) -> list:
    """Copy of the trials with a short handle (T01, T02, ...) on each."""
    return [{"candidate_id": f"T{i + 1:02d}", **t} for i, t in enumerate(trials)]


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def ground_ranked_trials(ranked: list, trials: list) -> tuple[list, list]:
    """
    Tie every entry Claude returned back to a real candidate.

    Claude gives three signals per entry: candidate_id, nct_id and title.
    Each one is looked up in the list we SENT. The candidate that most signals
    agree on wins (ties: candidate_id > nct_id > title). nct_id and title are
    then overwritten from our own data, so whatever reaches the UI is
    guaranteed to be a retrieved trial with a working link and a SHAP row.
    Entries that match nothing are dropped and returned separately.
    """
    by_cid   = {f"T{i + 1:02d}": i for i in range(len(trials))}
    by_nct   = {str(t.get("nct_id", "")).strip().upper(): i for i, t in enumerate(trials)}
    by_title = {_norm(t.get("title")): i for i, t in enumerate(trials)}

    grounded, dropped, used = [], [], set()
    for entry in ranked:
        if not isinstance(entry, dict):
            continue
        signals = [
            by_cid.get(str(entry.get("candidate_id", "")).strip().upper()),
            by_nct.get(str(entry.get("nct_id", "")).strip().upper()),
            by_title.get(_norm(entry.get("title"))),
        ]
        votes = [s for s in signals if s is not None]
        if not votes:
            dropped.append({"nct_id": entry.get("nct_id"), "title": entry.get("title"),
                            "reason": "matched no retrieved trial"})
            continue
        idx = max(votes, key=lambda s: (votes.count(s), -signals.index(s)))
        if idx in used:
            dropped.append({"nct_id": entry.get("nct_id"), "title": entry.get("title"),
                            "reason": "duplicate of a higher-ranked entry"})
            continue
        used.add(idx)

        real = trials[idx]
        if str(entry.get("nct_id", "")).strip().upper() != str(real.get("nct_id", "")).strip().upper():
            print(f"[AGENT] Corrected NCT ID {entry.get('nct_id')} -> {real.get('nct_id')} "
                  f"({str(real.get('title'))[:50]})")
        entry["nct_id"] = real.get("nct_id")
        entry["title"]  = real.get("title")
        grounded.append(entry)

    for new_rank, entry in enumerate(grounded, start=1):   # close gaps left by drops
        entry["rank"] = new_rank
    return grounded, dropped


def _parse_json_array(raw: str) -> list:
    """Pull the JSON array out of the reply even if there's stray text around it."""
    clean = raw.replace("```json", "").replace("```", "").strip()
    start, end = clean.find("["), clean.rfind("]")
    if start != -1 and end > start:
        clean = clean[start:end + 1]
    return json.loads(clean)


def build_user_message(patient: dict, trials: list) -> str:
    """
    Construct the messsage sent to the agent with patient data and trial candidates.
    """
    return f"""
## Patient Profile
{json.dumps(patient, indent=2)}

## Clinical Trial Candidates ({len(trials)} trials retrieved from ClinicalTrials.gov)
{json.dumps(tag_candidates(trials), indent=2)}

Please evaluate these trials against the patient profile using the clinical hierarchy.
Return your ranked top 5-10 matches as a JSON array.
"""


def run_agent(patient: dict, trials: list) -> list:
    """
    Send patient + trials to Claude and get back ranked recommendations.
    This is the core agentic reasoning step. 
    """
    if not trials:
        print("[AGENT] No trials to evaluate.")
        return []
    
    print(f"[AGENT] Reasoning over {len(trials)} trial candidates for patient {patient['patient_id']}...")

    user_message = build_user_message(patient, trials)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    raw_output = response.content[0].text

    # Parse, then ground every entry against the list we actually sent
    try:
        parsed = _parse_json_array(raw_output)
    except json.JSONDecodeError:
        print("[AGENT] Warning: Could not parse JSON. Raw output:")
        print(raw_output)
        return []

    ranked_trials, dropped = ground_ranked_trials(parsed, trials)
    print(f"[AGENT] Ranked {len(ranked_trials)} matching trials "
          f"({len(dropped)} removed as unverifiable).")

    # Write a small debug record so any mismatch can be diagnosed afterwards
    try:
        with open("last_agent_run.json", "w") as f:
            json.dump({
                "sent_nct_ids":     [t.get("nct_id") for t in trials],
                "returned_raw":     [{k: e.get(k) for k in ("candidate_id", "nct_id", "title")}
                                     for e in parsed if isinstance(e, dict)],
                "kept_nct_ids":     [e.get("nct_id") for e in ranked_trials],
                "dropped":          dropped,
                "stop_reason":      response.stop_reason,
            }, f, indent=2)
    except OSError:
        pass

    return ranked_trials


if __name__ == "__main__":
    from synthetic_patients import generate_patient
    from trials_api import fetch_trials

    patient = generate_patient(1)
    trials = fetch_trials(patient)
    results = run_agent(patient, trials)

    print("\n=== TOP MATCHES ===")
    print(json.dumps(results, indent=2))
