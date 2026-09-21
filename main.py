"""
main.py
Entry point for the Clinical Trial Matching Agent.

Workflow:
  1. Generate (or load) a patient record
  2. Agent determines what to search for based on patient profile
  3. Fetch live trials from ClinicalTrials.gov API
  4. Claude agent reasons through eligibility and ranks matches
  5. Print results with explanations

Usage:
  python main.py
"""

import json
from synthetic_patients import generate_patient_cohort
from trials_api import fetch_trials
from agent import run_agent

def run_for_patient(patient: dict):
    """Full pipeline for a single patient."""
    print("\n" + "=" * 60)
    print(f"PATIENT: {patient['patient_id']}")
    print(f" Age: {patient['age']}   |   Sex: {patient['sex']}")
    print(f" Condition: {patient['diagnosed_condition']}")
    print(f" Stage: {patient['disease_stage']}")
    print(f" ECOG: {patient['ecog_performance_status']}")
    print(f" Prior Treatments: {', '.join(patient['prior_treatments'])}")
    print("=" * 60)


    # Step 1: Fetch relevant trials from ClinicalTrials.gov
    trials = fetch_trials(patient)

    if not trials:
        print("  No trials retrieved from API. Try adjusting the search parameters.")
        return None
    
    # Step 2: Run Claude agent to reason through and rank the trials
    ranked_results = run_agent(patient, trials)

    # Step 3: display results
    if ranked_results:
        print(f"\n TOP {len(ranked_results)} MATCHED TRIALS:\n")
        for match in ranked_results:
            print(f"  #{match.get('rank', '?')} [{match.get('match_score', 'N/A')}] {match.get('title', 'N/A')}")
            print(f"     NCT ID : {match.get('nct_id', 'N/A')}")
            print(f".    Why    : {match.get('why_matched', 'N/A')}")

            flags = match.get("eligibility_flags", [])
            if flags:
                print(f"  ⚠ Flags :  {' | '.join(flags)}")
            
            notes = match.get("ambiguity_notes", [])
            if notes:
                print(f"  ~ Notes : {' | '.join(notes)}")
            
            print(f"     Next     : {match.get('recommended_next_step', 'N/A')}")
            print()
    else:
        print("  Agent returned no ranked matches.")

    return ranked_results


def main():
    # Generate a small cohort of synthetic patients
    print("Generating synthetic patient cohort...")
    cohort = generate_patient_cohort(n=2) # Start with 2 patient for demo

    all_results = {}

    for patient in cohort:
        results = run_for_patient(patient)
        if results:
            all_results[patient["patient_id"]] = {
                "patient": patient,
                "matched_trials": results
            }

        # Optionally save full results to JSON
        with open("results.json", "w") as f:
            json.dump(all_results, f, indent=2)
            print("\n Fulll results saved to results.json")


if __name__ == "__main__":
    main()
