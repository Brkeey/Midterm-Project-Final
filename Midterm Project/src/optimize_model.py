from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS = [
    "Avg_F0_Hz", "Avg_ZCR_per_s", "Avg_Energy", "Voiced_Frame_Ratio",
    "Spectral_Centroid_Mean", "Spectral_Bandwidth_Mean",
    "Spectral_Rolloff_Mean", "Spectral_Flatness_Mean",
    "MFCC1_Mean", "MFCC2_Mean", "MFCC3_Mean", "MFCC4_Mean", "MFCC5_Mean",
]
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]


def _norm(v: object) -> str:
    s = unicodedata.normalize("NFKD", str(v).strip().lower())
    return s.encode("ascii", "ignore").decode("ascii")


def _map_emotion(v: object, fname: object = "") -> str | None:
    table = {
        "angry": "Angry", "ofkeli": "Angry", "furious": "Angry",
        "happy": "Happy", "mutlu": "Happy",
        "neutral": "Neutral", "notr": "Neutral", "ntr": "Neutral",
        "sad": "Sad", "uzgun": "Sad", "mutsuz": "Sad",
        "surprised": "Surprised", "saskin": "Surprised", "shocked": "Surprised",
    }
    direct = table.get(_norm(v))
    if direct:
        return direct
    fn = _norm(fname).replace(".wav", "")
    for p in reversed(fn.split("_")[-5:]):
        if p in table:
            return table[p]
    return None


def optimize_gender_thresholds(feat_df: pd.DataFrame) -> tuple[float, float, float]:
    """Grid search over F0 cut-points for Male/Female/Child classification."""
    df = feat_df[
        feat_df["Avg_F0_Hz"].notna() & feat_df["Gender"].isin(["Male", "Female", "Child"])
    ].copy()

    best_acc, best_mf, best_fc = 0.0, 170.0, 255.0
    for mf in np.arange(140, 200, 2):
        for fc in np.arange(230, 290, 2):
            if mf >= fc:
                continue
            preds = df["Avg_F0_Hz"].apply(
                lambda f, m=mf, c=fc: "Male" if f < m else ("Female" if f < c else "Child")
            )
            acc = float((preds == df["Gender"]).mean())
            if acc > best_acc:
                best_acc, best_mf, best_fc = acc, float(mf), float(fc)

    return best_mf, best_fc, best_acc * 100


def optimize_emotion_classifier(
    df: pd.DataFrame, output_dir: Path
) -> tuple[dict, object | None, list[str]]:
    """Run GridSearchCV across three classifier families; return comparison results and best model."""
    df = df.copy()
    df["Emotion"] = df.apply(
        lambda r: _map_emotion(r.get("Feeling", ""), r.get("File_Name", "")), axis=1
    )
    df = df.dropna(subset=["Emotion"])
    df = df[df["Emotion"].isin(EMOTIONS)]

    available = [c for c in FEATURE_COLS if c in df.columns and df[c].notna().any()]
    X, y = df[available], df["Emotion"]

    if len(df) < 50:
        raise ValueError("Not enough labeled samples for optimization.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates: dict[str, tuple] = {
        "GradientBoosting": (
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
                "clf__subsample": [0.8, 1.0],
            },
        ),
        "RandomForest": (
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
            ]),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 20],
                "clf__min_samples_split": [2, 5],
                "clf__max_features": ["sqrt", "log2"],
            },
        ),
        "MLP": (
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("clf", MLPClassifier(random_state=42, max_iter=600)),
            ]),
            {
                "clf__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
                "clf__alpha": [0.001, 0.01],
                "clf__learning_rate_init": [0.001, 0.01],
            },
        ),
    }

    results: dict[str, dict] = {}
    best_name, best_score, best_estimator = None, 0.0, None

    for name, (pipe, params) in candidates.items():
        print(f"  [{name}] Searching {len(params)} param groups × 5 folds ...")
        gs = GridSearchCV(pipe, params, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
        gs.fit(X, y)
        acc = gs.best_score_ * 100
        results[name] = {
            "cv_accuracy": round(acc, 2),
            "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        }
        print(f"  [{name}] CV accuracy: {acc:.2f}%")
        if acc > best_score:
            best_score, best_name, best_estimator = acc, name, gs.best_estimator_

    if best_estimator is not None:
        model_path = output_dir / "emotion_model.joblib"
        joblib.dump({"model": best_estimator, "feature_cols": available}, model_path)
        print(f"\n  Saved best model ({best_name}) -> {model_path}")

    return results, best_estimator, available


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3: Optimize gender thresholds + emotion classifier.")
    p.add_argument("--features", type=Path, default=ROOT / "outputs" / "features.csv")
    p.add_argument("--metadata", type=Path, default=ROOT / "data" / "master_metadata.xlsx")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading data...")
    feat_df = pd.read_csv(args.features)
    meta_df = pd.read_excel(args.metadata)

    # ── Gender threshold optimization ──────────────────────────────────────────
    print("\n=== Gender F0 Threshold Optimization ===")
    mf_thresh, fc_thresh, opt_gender_acc = optimize_gender_thresholds(feat_df)

    df_g = feat_df[
        feat_df["Avg_F0_Hz"].notna() & feat_df["Gender"].isin(["Male", "Female", "Child"])
    ].copy()
    baseline_preds = df_g["Avg_F0_Hz"].apply(
        lambda f: "Male" if f < 170 else ("Female" if f < 255 else "Child")
    )
    baseline_gender_acc = float((baseline_preds == df_g["Gender"]).mean()) * 100

    print(f"  Baseline   : {baseline_gender_acc:.2f}%  (Male<170, Female<255)")
    print(f"  Optimized  : {opt_gender_acc:.2f}%  (Male<{mf_thresh}, Female<{fc_thresh})")

    # ── Emotion classifier optimization ────────────────────────────────────────
    print("\n=== Emotion Classifier Optimization ===")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = feat_df.merge(meta_df[["File_Name", "Feeling"]], on="File_Name", how="left")
    emotion_results, _, feature_cols = optimize_emotion_classifier(merged, args.output_dir)

    best_emotion_name = max(emotion_results, key=lambda k: emotion_results[k]["cv_accuracy"])
    best_emotion_acc = emotion_results[best_emotion_name]["cv_accuracy"]

    # ── Save JSON report (loaded by the Streamlit app) ─────────────────────────
    report = {
        "gender": {
            "baseline_accuracy": round(baseline_gender_acc, 2),
            "optimized_accuracy": round(opt_gender_acc, 2),
            "optimal_male_female_threshold": mf_thresh,
            "optimal_female_child_threshold": fc_thresh,
        },
        "emotion": {
            "baseline_accuracy": 43.05,
            "models": emotion_results,
            "best_model": best_emotion_name,
            "best_cv_accuracy": round(best_emotion_acc, 2),
            "feature_cols": feature_cols,
        },
    }
    report_path = args.output_dir / "optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report -> {report_path}")

    # ── Human-readable summary ─────────────────────────────────────────────────
    summary_lines = [
        "=== Phase 3 Optimization Report ===",
        "",
        "Gender Classifier (Rule-based F0 Thresholds):",
        f"  Baseline  : {baseline_gender_acc:.2f}%  (Male<170 Hz | Female<255 Hz)",
        f"  Optimized : {opt_gender_acc:.2f}%  (Male<{mf_thresh} Hz | Female<{fc_thresh} Hz)",
        "",
        "Emotion Classifier (ML, 5-fold CV):",
        f"  Baseline (GBT default): 43.05%",
    ]
    for name, res in emotion_results.items():
        marker = " <-- BEST" if name == best_emotion_name else ""
        summary_lines.append(f"  {name:20s}: {res['cv_accuracy']:.2f}%{marker}")
    summary_lines += [
        "",
        f"Best emotion model : {best_emotion_name} ({best_emotion_acc:.2f}% CV accuracy)",
        f"Features used      : {', '.join(feature_cols)}",
    ]
    summary = "\n".join(summary_lines)
    (args.output_dir / "optimization_summary.txt").write_text(summary)
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
