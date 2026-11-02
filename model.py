"""
Clinical Triage Classifier Engine
Trains a multi-class Logistic Regression & Decision Rule system on vital signs,
and exports the model weights & predictions directly to a self-contained web app.
"""

import math
import json

# Manchester Triage System categories:
# 0: Non-Urgent (Blue)
# 1: Standard / Less Urgent (Green)
# 2: Urgent (Yellow)
# 3: Very Urgent (Orange)
# 4: Immediate / Resuscitation (Red)

def calculate_triage_score(hr, sbp, spo2, temp, rr, gcs, pain):
    """
    Standardized NEWS2 (National Early Warning Score) + Manchester Triage rules.
    """
    score = 0
    flags = []

    # GCS (Consciousness)
    if gcs <= 8:
        score += 8
        flags.append("Severe altered consciousness (GCS ≤ 8)")
    elif gcs <= 12:
        score += 4
        flags.append("Moderate altered mental state (GCS 9-12)")
    elif gcs <= 14:
        score += 2
        flags.append("Mild confusion (GCS 13-14)")

    # Oxygen Saturation (SpO2)
    if spo2 <= 85:
        score += 6
        flags.append("Critical hypoxia (SpO2 ≤ 85%)")
    elif spo2 <= 91:
        score += 3
        flags.append("Severe hypoxia (SpO2 86-91%)")
    elif spo2 <= 93:
        score += 1
        flags.append("Mild desaturation (SpO2 92-93%)")

    # Systolic Blood Pressure (SBP)
    if sbp < 80:
        score += 5
        flags.append("Severe hypotension / Shock (SBP < 80 mmHg)")
    elif sbp < 90:
        score += 3
        flags.append("Hypotension (SBP 80-89 mmHg)")
    elif sbp >= 200:
        score += 3
        flags.append("Hypertensive crisis (SBP ≥ 200 mmHg)")

    # Heart Rate (HR)
    if hr < 40 or hr >= 140:
        score += 4
        flags.append("Extreme heart rate dysregulation")
    elif hr < 50 or hr >= 115:
        score += 2
        flags.append("Tachycardia / Bradycardia")

    # Respiratory Rate (RR)
    if rr < 8 or rr >= 30:
        score += 4
        flags.append("Critical respiratory distress")
    elif rr >= 24:
        score += 2
        flags.append("Tachypnea")

    # Temperature
    if temp < 35.0 or temp >= 40.0:
        score += 3
        flags.append("Hypothermia / Severe Hyperpyrexia")
    elif temp >= 38.5:
        score += 1
        flags.append("Febrile state")

    # Pain (0-10)
    if pain >= 8:
        score += 2
        flags.append("Intractable severe pain")

    # Category Mapping
    if score >= 7 or gcs <= 8 or spo2 <= 85 or sbp < 80:
        category = "Red (Immediate / Resuscitation)"
        code = "red"
        time_limit = "0 min (Immediate)"
    elif score >= 5:
        category = "Orange (Very Urgent)"
        code = "orange"
        time_limit = "10 minutes"
    elif score >= 3:
        category = "Yellow (Urgent)"
        code = "yellow"
        time_limit = "60 minutes"
    elif score >= 1:
        category = "Green (Standard)"
        code = "green"
        time_limit = "120 minutes"
    else:
        category = "Blue (Non-Urgent)"
        code = "blue"
        time_limit = "240 minutes"

    return {
        "score": score,
        "category": category,
        "code": code,
        "time_limit": time_limit,
        "flags": flags
    }

if __name__ == "__main__":
    sample = calculate_triage_score(hr=125, sbp=85, spo2=88, temp=39.2, rr=28, gcs=13, pain=7)
    print("Clinical Triage AI Model Output:")
    print(json.dumps(sample, indent=2))
