"""
Unit tests for Clinical Triage rules (NEWS2 + Manchester Triage System).
Verifies normal vital sign baselines, boundary transitions, and critical red-flag triggers.
"""
import pytest
from model import calculate_triage_score


def test_normal_healthy_patient_is_blue():
    """Healthy baseline patient should receive score 0 and Blue (Non-Urgent) category."""
    res = calculate_triage_score(hr=75, sbp=120, spo2=98, temp=36.8, rr=16, gcs=15, pain=0)
    assert res["score"] == 0
    assert res["code"] == "blue"
    assert len(res["flags"]) == 0


def test_critical_coma_triggers_red_immediately():
    """GCS <= 8 must trigger immediate Red (Resuscitation) priority regardless of other vitals."""
    res = calculate_triage_score(hr=75, sbp=120, spo2=98, temp=36.8, rr=16, gcs=8, pain=0)
    assert res["code"] == "red"
    assert res["score"] >= 7
    assert any("GCS" in f for f in res["flags"])


def test_critical_hypoxia_triggers_red_immediately():
    """SpO2 <= 85% must trigger immediate Red priority."""
    res = calculate_triage_score(hr=80, sbp=120, spo2=84, temp=36.8, rr=18, gcs=15, pain=2)
    assert res["code"] == "red"
    assert any("hypoxia" in f.lower() for f in res["flags"])


def test_severe_hypotension_shock_triggers_red():
    """Systolic BP < 80 mmHg indicates hemodynamic shock and must trigger Red priority."""
    res = calculate_triage_score(hr=85, sbp=75, spo2=96, temp=36.8, rr=18, gcs=15, pain=2)
    assert res["code"] == "red"
    assert any("hypotension" in f.lower() for f in res["flags"])


def test_orange_very_urgent_threshold():
    """Combined moderate abnormalities (score 5-6 without single critical red trigger) map to Orange."""
    # SpO2=90 (+3), HR=120 (+2) -> score 5 -> Orange
    res = calculate_triage_score(hr=120, sbp=115, spo2=90, temp=37.0, rr=18, gcs=15, pain=3)
    assert res["score"] == 5
    assert res["code"] == "orange"


def test_yellow_urgent_threshold():
    """Moderate abnormality (score 3-4) maps to Yellow (Urgent)."""
    # SBP=85 (+3) -> score 3 -> Yellow
    res = calculate_triage_score(hr=82, sbp=85, spo2=96, temp=37.0, rr=18, gcs=15, pain=2)
    assert res["score"] == 3
    assert res["code"] == "yellow"


def test_green_standard_threshold():
    """Mild single abnormality (score 1-2) maps to Green (Standard)."""
    # Fever 38.8 (+1) -> score 1 -> Green
    res = calculate_triage_score(hr=80, sbp=120, spo2=97, temp=38.8, rr=16, gcs=15, pain=2)
    assert res["score"] == 1
    assert res["code"] == "green"

