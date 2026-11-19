"""
Optional Flask REST API server for Clinical Triage AI.
For interactive web UI on Streamlit Cloud, run `streamlit run app.py`.
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from model import calculate_triage_score

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')

MODEL_PATH = 'triage_model.pkl'
SCALER_PATH = 'scaler.pkl'

CATEGORY_NAMES = {
    0: {"name": "Non-Urgent (Blue)", "code": "blue", "time": "≤ 240 mins", "priority": 5},
    1: {"name": "Standard / Less Urgent (Green)", "code": "green", "time": "≤ 120 mins", "priority": 4},
    2: {"name": "Urgent (Yellow)", "code": "yellow", "time": "≤ 60 mins", "priority": 3},
    3: {"name": "Very Urgent (Orange)", "code": "orange", "time": "≤ 10 mins", "priority": 2},
    4: {"name": "Immediate / Resuscitation (Red)", "code": "red", "time": "0 mins (Immediate)", "priority": 1}
}


def train_and_save_model():
    np.random.seed(42)
    n_samples = 6000
    hr = np.random.normal(82, 22, n_samples).clip(35, 180)
    sbp = np.random.normal(122, 24, n_samples).clip(60, 230)
    spo2 = np.random.normal(97, 4, n_samples).clip(70, 100)
    rr = np.random.normal(17, 6, n_samples).clip(8, 45)
    temp = np.random.normal(36.8, 1.1, n_samples).clip(34.0, 41.5)
    gcs = np.random.choice([15, 14, 13, 12, 11, 10, 9, 8, 6, 3], size=n_samples,
                           p=[0.75, 0.08, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    pain = np.random.randint(0, 11, n_samples)

    X = np.column_stack([hr, sbp, spo2, rr, temp, gcs, pain])
    y = np.zeros(n_samples, dtype=int)
    code_to_idx = {"blue": 0, "green": 1, "yellow": 2, "orange": 3, "red": 4}
    for i in range(n_samples):
        res = calculate_triage_score(hr[i], sbp[i], spo2[i], temp[i], rr[i], gcs[i], pain[i])
        y[i] = code_to_idx[res["code"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_scaled, y)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return clf, scaler


def load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
        except Exception:
            pass
    return train_and_save_model()


model, scaler = load_or_train()


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
        pred_class = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0].tolist()
        category_info = CATEGORY_NAMES.get(pred_class, CATEGORY_NAMES[0])
        rule_res = calculate_triage_score(hr, sbp, spo2, temp, rr, gcs, pain)
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
                "flags": rule_res["flags"]
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
