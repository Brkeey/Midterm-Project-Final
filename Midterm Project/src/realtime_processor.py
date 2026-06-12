from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from extract_features import extract_file_features
from classifier import predict_emotion, predict_gender_rule_based


def _predict_from_features(feat: dict) -> dict:
    def _f(k: str) -> float:
        v = feat.get(k, float("nan"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    f0 = _f("Avg_F0_Hz")
    zcr = _f("Avg_ZCR_per_s")
    eng = _f("Avg_Energy")
    vr = _f("Voiced_Frame_Ratio")
    sc = _f("Spectral_Centroid_Mean")
    sb = _f("Spectral_Bandwidth_Mean")
    sr_ = _f("Spectral_Rolloff_Mean")
    sf_ = _f("Spectral_Flatness_Mean")
    m1, m2, m3, m4, m5 = (
        _f("MFCC1_Mean"), _f("MFCC2_Mean"), _f("MFCC3_Mean"),
        _f("MFCC4_Mean"), _f("MFCC5_Mean"),
    )
    return {
        **feat,
        "Predicted_Gender": predict_gender_rule_based(f0),
        "Predicted_Emotion": predict_emotion(f0, zcr, eng, vr, sc, sb, sr_, sf_, m1, m2, m3, m4, m5),
    }


def analyze_audio_array(y: np.ndarray, sr: int) -> dict:
    """Extract features and predict from a raw audio array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        sf.write(str(tmp), y, sr)
        return _predict_from_features(extract_file_features(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def analyze_audio_bytes(wav_bytes: bytes) -> tuple[dict, np.ndarray, int]:
    """Analyze WAV bytes (from file upload). Returns (result, audio_array, sample_rate)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = Path(f.name)
    try:
        y, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        result = _predict_from_features(extract_file_features(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    return result, y, int(sr)


def record_audio(duration: float = 3.0, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Record audio from the default microphone."""
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten(), sr
