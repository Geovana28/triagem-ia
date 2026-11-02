import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')

MODEL_PATH = 'triage_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Triage Categories (Manchester Triage System):
# 0: Blue (Non-Urgent)
# 1: Green (Standard)
# 2: Yellow (Urgent)
# 3: Orange (Very Urgent)
# 4: Red (Immediate / Resuscitation)

CATEGORY_NAMES = {
    0: {"name": "Non-Urgent (Blue)", "code": "blue", "time": "≤ 240 mins", "priority": 5},
    1: {"name": "Standard / Less Urgent (Green)", "code": "green", "time": "≤ 120 mins", "priority": 4},
    2: {"name": "Urgent (Yellow)", "code": "yellow", "time": "≤ 60 mins", "priority": 3},
    3: {"name": "Very Urgent (Orange)", "code": "orange", "time": "≤ 10 mins", "priority": 2},
    4: {"name": "Immediate / Resuscitation (Red)", "code": "red", "time": "0 mins (Immediate)", "priority": 1}
}

def train_and_save_model():
    """Generates synthetic clinical triage training dataset following NEWS2 + Manchester guidelines and trains a Random Forest classifier."""
    np.random.seed(42)
    N_SAMPLES = 6000

    # Vital sign distributions
    hr = np.random.normal(82, 22, N_SAMPLES).clip(35, 180)
    sbp = np.random.normal(122, 24, N_SAMPLES).clip(60, 230)
    spo2 = np.random.normal(97, 4, N_SAMPLES).clip(70, 100)
    rr = np.random.normal(17, 6, N_SAMPLES).clip(8, 45)
    temp = np.random.normal(36.8, 1.1, N_SAMPLES).clip(34.0, 41.5)
    gcs = np.random.choice([15, 14, 13, 12, 11, 10, 9, 8, 6, 3], size=N_SAMPLES, p=[0.75, 0.08, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    pain = np.random.randint(0, 11, N_SAMPLES)

    X = np.column_stack([hr, sbp, spo2, rr, temp, gcs, pain])
    y = np.zeros(N_SAMPLES, dtype=int)

    for i in range(N_SAMPLES):
        score = 0
        # GCS
        if gcs[i] <= 8: score += 8
        elif gcs[i] <= 12: score += 4
        elif gcs[i] <= 14: score += 2

        # SpO2
        if spo2[i] <= 85: score += 6
        elif spo2[i] <= 91: score += 3
        elif spo2[i] <= 93: score += 1

        # SBP
        if sbp[i] < 80: score += 5
        elif sbp[i] < 90 or sbp[i] >= 200: score += 3

        # HR
        if hr[i] < 40 or hr[i] >= 140: score += 4
        elif hr[i] < 50 or hr[i] >= 115: score += 2

        # RR
        if rr[i] < 8 or rr[i] >= 30: score += 4
        elif rr[i] >= 24: score += 2

        # Temp
        if temp[i] < 35.0 or temp[i] >= 40.0: score += 3
        elif temp[i] >= 38.5: score += 1

        # Pain
        if pain[i] >= 8: score += 2

        # Manchester Class Label
        if score >= 7 or gcs[i] <= 8 or spo2[i] <= 85 or sbp[i] < 80:
            y[i] = 4 # Red
        elif score >= 5:
            y[i] = 3 # Orange
        elif score >= 3:
            y[i] = 2 # Yellow
        elif score >= 1:
            y[i] = 1 # Green
        else:
            y[i] = 0 # Blue

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_scaled, y)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Scikit-Learn Triage Model & Scaler successfully trained and saved!")

# Ensure model exists
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    train_and_save_model()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        hr = float(data.get('hr', 75))
        sbp = float(data.get('sbp', 120))
        spo2 = float(data.get('spo2', 98))
        rr = float(data.get('rr', 16))
        temp = float(data.get('temp', 36.8))
        gcs = float(data.get('gcs', 15))
        pain = float(data.get('pain', 0))

        raw_features = np.array([[hr, sbp, spo2, rr, temp, gcs, pain]])
        features_scaled = scaler.transform(raw_features)

        # Scikit-Learn Model Prediction & Probabilities
        pred_class = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0].tolist()

        category_info = CATEGORY_NAMES.get(pred_class, CATEGORY_NAMES[0])

        # Flag detection
        flags = []
        if gcs <= 8: flags.append(f"Critical coma/unconsciousness (GCS {int(gcs)})")
        elif gcs <= 12: flags.append(f"Altered mental status (GCS {int(gcs)})")
        if spo2 <= 85: flags.append(f"Severe critical hypoxia (SpO2 {int(spo2)}%)")
        elif spo2 <= 91: flags.append(f"Desaturation (SpO2 {int(spo2)}%)")
        if sbp < 80: flags.append(f"Severe hypotension / shock state (SBP {int(sbp)} mmHg)")
        elif sbp >= 200: flags.append(f"Severe hypertensive crisis (SBP {int(sbp)} mmHg)")
        if hr >= 130 or hr < 45: flags.append(f"Severe heart rate dysregulation ({int(hr)} bpm)")
        if rr >= 28 or rr < 9: flags.append(f"Acute respiratory distress ({int(rr)} /min)")
        if temp >= 39.5 or temp < 35.0: flags.append(f"Extreme core temperature ({temp:.1f} °C)")
        if pain >= 8: flags.append(f"Severe intractable acute pain ({int(pain)}/10)")

        # Mortality / ICU risk estimate
        icu_prob = (probabilities[-1] * 0.7 + probabilities[-2] * 0.25) * 100.0

        return jsonify({
            "status": "success",
            "model": "Scikit-Learn Random Forest Classifier (100 estimators)",
            "prediction": {
                "class_index": pred_class,
                "category": category_info["name"],
                "code": category_info["code"],
                "priority": category_info["priority"],
                "time_target": category_info["time"],
                "icu_risk": round(icu_prob, 1),
                "probabilities": {
                    "blue": round(probabilities[0] * 100, 1),
                    "green": round(probabilities[1] * 100, 1) if len(probabilities) > 1 else 0,
                    "yellow": round(probabilities[2] * 100, 1) if len(probabilities) > 2 else 0,
                    "orange": round(probabilities[3] * 100, 1) if len(probabilities) > 3 else 0,
                    "red": round(probabilities[4] * 100, 1) if len(probabilities) > 4 else 0,
                },
                "flags": flags
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    print("Starting Clinical Triage ML Server on http://localhost:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
