"""
synthetic_patients.py
Generates realistic patient records with clinical relevant features 
for clinical matching
"""

import random
from datetime import datetime, timedelta

# --- Reference Data ---

CONDITIONS = [
    "Type 2 Diabetes", "Non-Small Cell Lung Cancer", "Breast Cancer",
    "Hypertension", "Chronic Kidney Disease", "Alzheimer's Disease",
    "Rheumatoid Arthritis", "Multiple Sclerosis", "Heart Failure",
    "Melanoma"
]

PRIOR_TREATMENTS = {
    "Type 2 Diabetes": ["Metformin", "Insulin", "GLP-1 agonist", "SGLT2 inhibitor"],
    "Non-Small Cell Lung Cancer": ["Carboplatin", "Pembrolizumab", "Erlotinib", "Radiation"],
    "Breast Cancer": ["Tamoxifen", "Herceptin", "Doxorubicin", "Radiation"],
    "Hypertension": ["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide"],
    "Chronic Kidney Disease": ["ACE inhibitors", "Erythropoietin", "Phosphate binders"],
    "Alzheimer's Disease": ["Donepezil", "Memantine", "Rivastigmine"],
    "Rheumatoid Arthritis": ["Methotrexate", "Adalimumab", "Prednisone", "Hydroxychloroquine"],
    "Multiple Sclerosis": ["Interferon beta", "Natalizumab", "Ocrelizumab"],
    "Heart Failure": ["Furosemide", "Carvedilol", "Sacubitril/Valsartan", "Spironolactone"],
    "Melanoma": ["Nivolumab", "Ipilimumab", "Dabrafenib", "Vemurafenib"]   
}

ENTHNICITIES = ["White", "Black or African American", "Hispanic or Latino", "Asian", "Other"]
ACTIVITY_LEVELS = ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
SMOKING_STATUS = ["Never", "Former", "Current"]
ECOG_STATUS = [0, 1, 2] # 0=fully active, 1=restricted, 2=ambulatory but limited


def generate_lab_values(condition):
    """Generate realistic lab values based on patient conditions."""
    labs = {
        # Complete Bloot Count
        "hemoglobin_g_dl": round(random.uniform(10.5, 16.5), 1),
        "wbc_k_ul": round(random.uniform(3.5, 11.0), 1),
        "platelets_k_ul": round(random.uniform(130, 400), 0),

        # Metabolic Panel
        "creatinine_mg_dl": round(random.uniform(0.6, 2.5), 2),
        "egfr_ml_min": round(random.uniform(30, 100), 0),
        "alt_u_l": round(random.uniform(10, 80), 0),
        "ast_u_l": round(random.uniform(10, 75), 0),
        "bilirubin_mg_dl": round(random.uniform(0.1, 1.5), 2),
        "albumin_g_dl": round(random.uniform(3.0, 5.0), 1),

        # Cardiac
        "ejection_fraction_pct": round(random.uniform(35, 70), 0),
        "ekg_qt_interval_ms": round(random.uniform(360, 480), 0),
        
        # Metabolic
        "hba1c_pct": round(random.uniform(5.0, 10.5), 1),
        "fasting_glucose_mg_dl": round(random.uniform(80, 280), 0),
        "ldl_mg_dl": round(random.uniform(60, 200), 0),
        "hdl_mg_dl": round(random.uniform(30, 80), 0),
        "triglycerides_mg_dl": round(random.uniform(80, 350), 0),

        # Oncology markers (if applicable)
        "psa_ng_ml": round(random.uniform(0.5, 15.0), 2),
        "ca125_u_ml": round(random.uniform(10, 100), 1),
    }

    # Adjust labs to be more realistic per condition
    if condition == "Chronic Kidney Disease":
        labs["creatinine_mg_dl"] = round(random.uniform(1.8, 4.5), 2)
        labs["egfr_ml_min"] = round(random.uniform(10, 45), 0)

    if condition == "Type 2 Diabetes":
        labs["hba1c_pct"] = round(random.uniform(7.5, 12.0), 1)
        labs["fasting_glucose_mg_dl"] = round(random.uniform(150, 320), 0)

    if condition == "Heart Failure":
        labs["ejection_fraction_pct"] = round(random.uniform(15, 45), 0)

    return labs


def generate_patient(patient_id):
    """Generate a single synthetic patient record"""
    age = random.randint(25, 80)
    condition = random.choice(CONDITIONS)
    treatments = PRIOR_TREATMENTS.get(condition, [])
    prior_tx = random.sample(treatments, k=random.randint(1, min(3, len(treatments))))

    # BMI
    height_cm = random.randint(155, 195)
    weight_kg = random.randint(55, 130)
    bmi = round(weight_kg / ((height_cm / 100) **2), 1)

    patient = {
        # --- TIER 1: Hard Filters ---
        "patient_id": f"PT-{patient_id:04d}",
        "age": age,
        "sex": random.choice(["Male", "Female"]),
        "diagnosed_condition": condition,
        "diagnosed_date": (datetime.today() - timedelta(days=random.randint(180, 3650))).strftime("%Y-%m-%d"),
        "disease_stage": random.choice(["Stage I", "Stage II", "Stage III", "Stage IV", "N/A"]),

        # --- TIER 2: Eligibility Validators ---
        "lab_values": generate_lab_values(condition),
        "prior_treatments": prior_tx,
        "allergies": random.sample(["Penicilin", "Sulfa", "NSAIDs", "Aspirin", "None"], k=random.randint(1,2)),
        "comorbidities": random.sample(
            ["Hypertension", "Type 2 Diabetes", "COPD", "Atrial Fibrillation", "Depression", "None"],
            k=random.randint(1,4)
        ),
        "current_medications": random.sample(
            ["Metformin", "Lisinopril", "Atorvastatin", "Omeprazole", "Levothyroxine"],
            k=random.randint(1,4)
        ),
        "ecog_performance_status": random.choice(ECOG_STATUS), # 0=best, 2=limited

        # --- TIER 3: Optimization/Ranking Factors ---
        "bmi":bmi,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "smoking_status": random.choice(SMOKING_STATUS),
        "alcohol_use": random.choice(["None", "Social", "Moderate", "Heavy"]),
        "activity_level": random.choice(ACTIVITY_LEVELS),
        "diet": random.choice(["Standard", "Vegetarian", "Vegan", "Low-Sodium","Diabetic"]),
        "ethnicity": random.choice(ENTHNICITIES),

        # --- Administrative ---
        "location_zip": f"{random.randint(10000, 99999)}",
        "insurance": random.choice(["Medicare", "Medicaid", "Private", "Uninsured"]),
        "willing_to_travel": random.choice([True, False]),
        "prior_trial_participation": random.choice([True, False]),
    }

    return patient


def generate_patient_cohort(n=10):
    """Generate a cohort of n synthetic patients."""
    return [generate_patient(i + 1) for i in range(n)]

if __name__== "__main__":
    import json
    cohort = generate_patient_cohort(3)
    print(json.dumps(cohort[0], indent=2)) # Preview first patient


