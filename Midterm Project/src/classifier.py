from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import joblib
import pandas as pd

<<<<<<< HEAD
_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _gender_thresholds() -> tuple[float, float]:
    """Return (male_female_thresh, female_child_thresh) from optimized report if available."""
    report_path = _ROOT / "outputs" / "optimization_report.json"
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text())
            g = data.get("gender", {})
            return (
                float(g.get("optimal_male_female_threshold", 170)),
                float(g.get("optimal_female_child_threshold", 255)),
            )
        except Exception:
            pass
    return 170.0, 255.0
=======
from feature_engineering import add_engineered_features
>>>>>>> 06ab68de5e479804017ced2b079bf7cc5cbda511


def predict_gender_rule_based(avg_f0_hz: float) -> str:
    """
    Pitch-based rule using optimized thresholds (falls back to 170/255 Hz defaults).
    - Male   : F0 < male_thresh
    - Female : male_thresh <= F0 < child_thresh
    - Child  : F0 >= child_thresh
    """
    if np.isnan(avg_f0_hz):
        return "Unknown"
    male_thresh, child_thresh = _gender_thresholds()
    if avg_f0_hz < male_thresh:
        return "Male"
    if avg_f0_hz < child_thresh:
        return "Female"
    return "Child"


def predict_emotion_rule_based(
    avg_f0_hz: float,
    avg_zcr_per_s: float,
    avg_energy: float,
    voiced_frame_ratio: float,
) -> str:
    """
    Simple heuristic emotion prediction from acoustic cues.
    Returns one of: Angry, Happy, Sad, Surprised, Neutral, Unknown.
    """
    values = [avg_f0_hz, avg_zcr_per_s, avg_energy, voiced_frame_ratio]
    if any(np.isnan(v) for v in values):
        return "Unknown"

    if avg_energy >= 0.020:
        if avg_f0_hz >= 220 or avg_zcr_per_s >= 1700:
            return "Surprised"
        return "Angry"

    if avg_f0_hz >= 190 and avg_zcr_per_s >= 1200:
        return "Happy"

    if avg_f0_hz <= 155 and avg_energy <= 0.008 and voiced_frame_ratio >= 0.60:
        return "Sad"

    return "Neutral"


@lru_cache(maxsize=1)
def _load_emotion_model() -> object | None:
    model_path = Path(__file__).resolve().parents[1] / "outputs" / "emotion_model.joblib"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def predict_emotion_from_row(feature_row: dict[str, float]) -> str:
    """Predict emotion using trained Phase-1/2 model and a feature dictionary."""
    avg_f0_hz = float(feature_row.get("Avg_F0_Hz", np.nan))
    avg_zcr_per_s = float(feature_row.get("Avg_ZCR_per_s", np.nan))
    avg_energy = float(feature_row.get("Avg_Energy", np.nan))
    voiced_frame_ratio = float(feature_row.get("Voiced_Frame_Ratio", np.nan))

    values = [avg_f0_hz, avg_zcr_per_s, avg_energy, voiced_frame_ratio]
    if any(np.isnan(v) for v in values):
        return "Unknown"

    model_bundle = _load_emotion_model()
    if model_bundle is not None:
        try:
            if isinstance(model_bundle, dict) and "model" in model_bundle:
                feature_cols = model_bundle.get("feature_cols", [])
                phase = int(model_bundle.get("phase", 1))
                frame = pd.DataFrame([feature_row])
                if phase >= 2:
                    frame = add_engineered_features(frame)
                x = frame.reindex(columns=feature_cols)
                pred = model_bundle["model"].predict(x)[0]
            else:
                pred = model_bundle.predict(pd.DataFrame([feature_row]))[0]
            return str(pred)
        except Exception:
            pass

    return predict_emotion_rule_based(avg_f0_hz, avg_zcr_per_s, avg_energy, voiced_frame_ratio)


def predict_emotion(
    avg_f0_hz: float,
    avg_zcr_per_s: float,
    avg_energy: float,
    voiced_frame_ratio: float,
    spectral_centroid_mean: float = np.nan,
    spectral_bandwidth_mean: float = np.nan,
    spectral_rolloff_mean: float = np.nan,
    spectral_flatness_mean: float = np.nan,
    mfcc1_mean: float = np.nan,
    mfcc2_mean: float = np.nan,
    mfcc3_mean: float = np.nan,
    mfcc4_mean: float = np.nan,
    mfcc5_mean: float = np.nan,
    **extra_features: float,
) -> str:
    """
    Predict emotion from acoustic features.
    Uses trained model when available, otherwise falls back to rule-based logic.
    """
    row: dict[str, float] = {
        "Avg_F0_Hz": avg_f0_hz,
        "Avg_ZCR_per_s": avg_zcr_per_s,
        "Avg_Energy": avg_energy,
        "Voiced_Frame_Ratio": voiced_frame_ratio,
        "Spectral_Centroid_Mean": spectral_centroid_mean,
        "Spectral_Bandwidth_Mean": spectral_bandwidth_mean,
        "Spectral_Rolloff_Mean": spectral_rolloff_mean,
        "Spectral_Flatness_Mean": spectral_flatness_mean,
        "MFCC1_Mean": mfcc1_mean,
        "MFCC2_Mean": mfcc2_mean,
        "MFCC3_Mean": mfcc3_mean,
        "MFCC4_Mean": mfcc4_mean,
        "MFCC5_Mean": mfcc5_mean,
    }
    row.update(extra_features)
    return predict_emotion_from_row(row)
