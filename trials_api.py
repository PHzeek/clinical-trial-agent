"""
trials_api.py
Handles all communication with the ClinicalTrials.gov API (v2).
Fetches live, recruiting clinical trials based on patient condition and criteria.

ClinicalTrials.gov API v2 docs: https://clinicaltrials.gov/data-api/api
No API key required.
"""

import re
import requests


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

FETCH_SIZE    = 50   # Over-fetch, because the Tier 1 guard below may drop some
MAX_CANDIDATES = 20  # What we hand to the scorer / agent / SHAP


def build_query(patient: dict) -> dict:
    """
    Build query parameters from patient data using our clinical hierarchy:
    Tier 1 -> condition, age, sex (hard filters)
    We let the agent handle deeper eligibility reasoning after retrieval.

    NOTE: API v2 has no `filter.ageRange` parameter. Unknown parameters are
    silently ignored, so it *looks* like it works while filtering nothing.
    Age and sex are filtered with an Essie expression in `filter.advanced`.
    """
    condition = patient.get("diagnosed_condition", "")
    age = patient.get("age")
    sex = patient.get("sex", "")

    advanced = []
    if isinstance(age, (int, float)) and age > 0:
        a = int(age)
        # Trial's minimum age must be <= patient age, maximum age must be >= it
        advanced.append(f"AREA[MinimumAge]RANGE[MIN, {a} years]")
        advanced.append(f"AREA[MaximumAge]RANGE[{a} years, MAX]")

    api_sex = {"Male": "MALE", "Female": "FEMALE"}.get(sex)
    if api_sex:
        advanced.append(f"(AREA[Sex]ALL OR AREA[Sex]{api_sex})")

    params = {
        "query.cond": condition,               # Filter by condition
        "filter.overallStatus": "RECRUITING",  # Only recruiting trials
        "pageSize": FETCH_SIZE,
        "format": "json",
    }
    if advanced:
        params["filter.advanced"] = " AND ".join(advanced)
    return params


# ── Tier 1 guard: never trust the server-side filter alone ────────────────

def _age_in_years(age_str) -> float | None:
    """'18 Years' -> 18.0, '6 Months' -> 0.5, 'N/A' / None -> None."""
    if not age_str:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)?", str(age_str))
    if not m:
        return None
    value, unit = float(m.group(1)), (m.group(2) or "years").lower()
    if unit.startswith("month"):
        return value / 12
    if unit.startswith("week"):
        return value / 52
    if unit.startswith("day"):
        return value / 365
    return value


def passes_tier1(patient: dict, trial: dict) -> bool:
    """True if the patient meets the trial's structured age and sex limits."""
    age = patient.get("age")
    if isinstance(age, (int, float)):
        lo = _age_in_years(trial.get("min_age"))
        hi = _age_in_years(trial.get("max_age"))
        if lo is not None and age < lo:
            return False
        if hi is not None and age > hi:
            return False

    trial_sex   = str(trial.get("sex", "ALL")).upper()
    patient_sex = str(patient.get("sex", "")).upper()
    if trial_sex not in ("ALL", "", "N/A") and patient_sex and trial_sex != patient_sex:
        return False
    return True


def _get(params: dict) -> list:
    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json().get("studies", [])


MIN_EXPECTED = 5   # Fewer than this from the filtered query -> double-check unfiltered


def fetch_trials(patient: dict) -> list:
    """
    Call the ClinicalTrials.gov API and return a clean list of trials the
    patient passes Tier 1 for (condition via query, age + sex verified locally).
    """
    params = build_query(patient)

    try:
        studies = _get(params)
    except requests.exceptions.RequestException as e:
        print(f"[API] Filtered query failed ({e}).")
        studies = []

    # Safety net: if the advanced filter errored or came back suspiciously thin,
    # pull the unfiltered condition search too. The local Tier 1 guard below
    # removes anything the patient doesn't qualify for either way.
    if len(studies) < MIN_EXPECTED and "filter.advanced" in params:
        print("[API] Few results with server-side age/sex filter - also checking unfiltered search...")
        fallback = {k: v for k, v in params.items() if k != "filter.advanced"}
        try:
            seen = {s.get("protocolSection", {}).get("identificationModule", {}).get("nctId") for s in studies}
            for s in _get(fallback):
                nid = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                if nid not in seen:
                    studies.append(s)
                    seen.add(nid)
        except requests.exceptions.RequestException as e:
            print(f"[API ERROR] Failed to fetch trials: {e}")
            if not studies:
                return []

    if not studies:
        print("[API] No studies returned. Try broadening the search.")
        return []

    # Flatten the nested API response into clean dicts
    trials = []
    for study in studies:
        proto        = study.get("protocolSection", {})
        id_mod       = proto.get("identificationModule", {})
        status_mod   = proto.get("statusModule", {})
        desc_mod     = proto.get("descriptionModule", {})
        cond_mod     = proto.get("conditionsModule", {})
        elig_mod     = proto.get("eligibilityModule", {})
        sponsor_mod  = proto.get("sponsorCollaboratorsModule", {})
        contacts_mod = proto.get("contactsLocationsModule", {})

        # Extract location info (first location if available)
        locations = contacts_mod.get("locations", [])
        city  = locations[0].get("city", "N/A") if locations else "N/A"
        state = locations[0].get("state", "N/A") if locations else "N/A"

        trials.append({
            "nct_id": id_mod.get("nctId", "N/A"),
            "title": id_mod.get("briefTitle", "N/A"),
            "condition": cond_mod.get("conditions", []),
            "summary": desc_mod.get("briefSummary", "N/A"),
            "eligibility_criteria": elig_mod.get("eligibilityCriteria", "N/A"),
            "min_age": elig_mod.get("minimumAge", "N/A"),
            "max_age": elig_mod.get("maximumAge", "N/A"),
            "sex": elig_mod.get("sex", "ALL"),
            "status": status_mod.get("overallStatus", "N/A"),
            "phase": proto.get("designModule", {}).get("phases", ["N/A"]),
            "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", "N/A"),
            "location_city": city,
            "location_state": state,
            "start_date": status_mod.get("startDateStruct", {}).get("date", "N/A"),
        })

    # Tier 1 guard
    eligible = [t for t in trials if passes_tier1(patient, t)]
    dropped  = len(trials) - len(eligible)
    if dropped:
        print(f"[API] Dropped {dropped} trial(s) failing Tier 1 age/sex limits.")
    eligible = eligible[:MAX_CANDIDATES]

    print(f"[API] Retrieved {len(eligible)} trials for condition: {patient['diagnosed_condition']}")
    return eligible


if __name__ == "__main__":
    import json
    from synthetic_patients import generate_patient

    patient = generate_patient(1)
    print(f"\nSearching trials for: {patient['diagnosed_condition']}, Age {patient['age']}, {patient['sex']}\n")
    print("Query params:", json.dumps(build_query(patient), indent=2))
    trials = fetch_trials(patient)

    if trials:
        print(json.dumps(trials[0], indent=2))  # Preview first result

