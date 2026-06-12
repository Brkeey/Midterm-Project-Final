from __future__ import annotations

import json
import tempfile
from pathlib import Path
<<<<<<< HEAD
=======
import pandas as pd
import streamlit as st
from classifier import predict_emotion_from_row, predict_gender_rule_based
from extract_features import resolve_audio_path
>>>>>>> 06ab68de5e479804017ced2b079bf7cc5cbda511

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from classifier import predict_emotion, predict_gender_rule_based
from realtime_processor import analyze_audio_array, analyze_audio_bytes, record_audio

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data" / "master_metadata.xlsx"
FEATURES_PATH = ROOT / "outputs" / "features_phase2.csv"
if not FEATURES_PATH.exists():
    FEATURES_PATH = ROOT / "outputs" / "features.csv"
SUMMARY_PATH = ROOT / "outputs" / "evaluation_summary.txt"
EMOTION_SUMMARY_PATH = ROOT / "outputs" / "emotion_model_summary.txt"
STATS_PATH = ROOT / "outputs" / "class_statistics.csv"
CM_PATH = ROOT / "outputs" / "confusion_matrix.csv"
EMOTION_CM_PATH = ROOT / "outputs" / "emotion_confusion_matrix.csv"
OPT_REPORT_PATH = ROOT / "outputs" / "optimization_report.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset():
    meta = pd.read_excel(META_PATH)
    feat = pd.read_csv(FEATURES_PATH)
    stats = pd.read_csv(STATS_PATH) if STATS_PATH.exists() else pd.DataFrame()
    cm = pd.read_csv(CM_PATH, index_col=0) if CM_PATH.exists() else pd.DataFrame()
    ecm = pd.read_csv(EMOTION_CM_PATH, index_col=0) if EMOTION_CM_PATH.exists() else pd.DataFrame()
    return meta, feat, stats, cm, ecm


<<<<<<< HEAD
def parse_accuracy(path: Path) -> str:
    if not path.exists():
        return "N/A"
    for line in path.read_text().splitlines():
        if "Overall Accuracy" in line or "Emotion Model Accuracy" in line:
=======
def parse_accuracy(summary_path: Path, key: str) -> str:
    if not summary_path.exists():
        return "N/A"
    txt = summary_path.read_text()
    for line in txt.splitlines():
        if key in line:
>>>>>>> 06ab68de5e479804017ced2b079bf7cc5cbda511
            return line.split(":")[-1].strip()
    return "N/A"


def load_opt_report() -> dict | None:
    if OPT_REPORT_PATH.exists():
        try:
            return json.loads(OPT_REPORT_PATH.read_text())
        except Exception:
            return None
    return None


def _plot_waveform(y: np.ndarray, sr: int, title: str = "Waveform") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2))
    t = np.linspace(0, len(y) / sr, len(y))
    if len(y) > 12000:
        step = len(y) // 12000
        t, y = t[::step], y[::step]
    ax.plot(t, y, linewidth=0.5, color="#1f6bae")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_cm(cm_df: pd.DataFrame, title: str) -> plt.Figure:
    vals = cm_df.values.astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(vals, cmap="Blues")
    ax.set_xticks(range(cm_df.shape[1]))
    ax.set_yticks(range(cm_df.shape[0]))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cm_df.index, fontsize=9)
    vmax = vals.max() if vals.max() > 0 else 1
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            v = int(vals[i, j])
            color = "white" if vals[i, j] > vmax * 0.6 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    return fig


def _show_analysis_result(result: dict, y: np.ndarray, sr: int, audio_bytes: bytes | None = None) -> None:
    """Render prediction metrics, waveform, and feature table for one audio sample."""
    gender = result.get("Predicted_Gender", "Unknown")
    emotion = result.get("Predicted_Emotion", "Unknown")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Gender", gender)
    c2.metric("Predicted Emotion", emotion)
    f0 = result.get("Avg_F0_Hz", float("nan"))
    f0_str = f"{float(f0):.1f} Hz" if pd.notna(f0) else "N/A"
    c3.metric("Average F0", f0_str)
    dur = result.get("Duration_s", 0)
    c4.metric("Duration", f"{float(dur):.2f} s")

    st.pyplot(_plot_waveform(y, sr, title="Audio Waveform"))

    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")

    with st.expander("Extracted Features", expanded=False):
        feature_keys = [
            "Sample_Rate", "Duration_s", "Avg_F0_Hz", "Avg_ZCR_per_s",
            "Avg_Energy", "Voiced_Frame_Ratio",
            "Spectral_Centroid_Mean", "Spectral_Bandwidth_Mean",
            "Spectral_Rolloff_Mean", "Spectral_Flatness_Mean",
            "MFCC1_Mean", "MFCC2_Mean", "MFCC3_Mean", "MFCC4_Mean", "MFCC5_Mean",
        ]
        rows = []
        for k in feature_keys:
            v = result.get(k)
            if v is not None:
                try:
                    rows.append({"Feature": k, "Value": f"{float(v):.4f}"})
                except (TypeError, ValueError):
                    rows.append({"Feature": k, "Value": str(v)})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Feature"), use_container_width=True)


# ── Page layout ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Speech Emotion & Gender Analyzer", layout="wide")
    st.title("Speech Emotion & Gender Analyzer — Phase 3")
    st.caption("Real-time analysis via microphone recording or file upload")

<<<<<<< HEAD
    meta, feat, stats, cm, ecm = load_dataset()
    opt = load_opt_report()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset Overview",
        "Upload & Analyze",
        "Live Recording",
        "Model Performance",
    ])
=======
    meta, feat, stats, cm = load_data()
    gender_accuracy = parse_accuracy(SUMMARY_PATH, "Overall Accuracy")
    emotion_accuracy = parse_accuracy(EMOTION_SUMMARY_PATH, "Emotion Model Accuracy")

    st.subheader("Dataset Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gender Accuracy (%)", gender_accuracy)
    c2.metric("Emotion Accuracy Phase-2 (%)", emotion_accuracy)
    c3.metric("Total Matched Files", int(meta["Audio_Exists"].sum()))
    c4.metric("Feature Rows", len(feat))
>>>>>>> 06ab68de5e479804017ced2b079bf7cc5cbda511

    # ── Tab 1: Dataset Overview ────────────────────────────────────────────────
    with tab1:
        gender_acc = parse_accuracy(SUMMARY_PATH)
        emotion_acc = parse_accuracy(ROOT / "outputs" / "emotion_model_summary.txt")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gender Accuracy (%)", gender_acc)
        c2.metric("Emotion Accuracy (%)", emotion_acc)
        c3.metric("Matched Audio Files", int(meta["Audio_Exists"].sum()))
        c4.metric("Feature Rows", len(feat))

<<<<<<< HEAD
        st.markdown("---")
        st.subheader("Single File Inspection")
        merged = feat.merge(meta[["File_Name", "Audio_Path", "Feeling"]], on="File_Name", how="left")
        options = merged["File_Name"].dropna().tolist()
        selected = st.selectbox("Select a file from the dataset", options)
=======
    row = merged[merged["File_Name"] == selected].iloc[0]
    numeric_cols = feat.select_dtypes(include="number").columns
    feature_row = {c: float(row[c]) for c in numeric_cols if c in row.index and pd.notna(row[c])}

    f0 = float(row["Avg_F0_Hz"]) if pd.notna(row.get("Avg_F0_Hz")) else float("nan")
    pred = predict_gender_rule_based(f0)
    pred_emotion = predict_emotion_from_row(feature_row)
>>>>>>> 06ab68de5e479804017ced2b079bf7cc5cbda511

        row = merged[merged["File_Name"] == selected].iloc[0]

<<<<<<< HEAD
        def _fv(col: str) -> float:
            v = row.get(col, float("nan"))
            return float(v) if pd.notna(v) else float("nan")
=======
    audio_path = resolve_audio_path(Path(str(row.get("Audio_Path", ""))), project_root=ROOT)
    if audio_path is not None:
        st.audio(str(audio_path))
    else:
        st.warning("Audio file not found for playback.")
>>>>>>> 06ab68de5e479804017ced2b079bf7cc5cbda511

        f0 = _fv("Avg_F0_Hz")
        pred = predict_gender_rule_based(f0)
        pred_emotion = predict_emotion(
            f0, _fv("Avg_ZCR_per_s"), _fv("Avg_Energy"), _fv("Voiced_Frame_Ratio"),
            _fv("Spectral_Centroid_Mean"), _fv("Spectral_Bandwidth_Mean"),
            _fv("Spectral_Rolloff_Mean"), _fv("Spectral_Flatness_Mean"),
            _fv("MFCC1_Mean"), _fv("MFCC2_Mean"), _fv("MFCC3_Mean"),
            _fv("MFCC4_Mean"), _fv("MFCC5_Mean"),
        )

        c4, c5, c6, c7, c8 = st.columns(5)
        c4.metric("Predicted Gender", pred)
        c5.metric("Actual Gender", str(row.get("Gender", "")))
        c6.metric("Predicted Emotion", pred_emotion)
        c7.metric("Actual Emotion", str(row.get("Feeling", "")))
        c8.metric("Avg F0 (Hz)", f"{f0:.1f}" if pd.notna(f0) else "N/A")

        audio_path = str(row.get("Audio_Path", ""))
        if audio_path and Path(audio_path).exists():
            st.audio(audio_path)
        else:
            st.warning("Audio file not found for playback.")

        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Class Statistics")
            if not stats.empty:
                st.dataframe(stats, use_container_width=True)
        with col_right:
            st.subheader("Gender Confusion Matrix")
            if not cm.empty:
                st.pyplot(_plot_cm(cm, "Gender Classifier"))

    # ── Tab 2: Upload & Analyze ────────────────────────────────────────────────
    with tab2:
        st.subheader("Upload a WAV File for Instant Analysis")
        uploaded = st.file_uploader("Choose a WAV audio file", type=["wav"], key="upload_wav")

        if uploaded is not None:
            wav_bytes = uploaded.read()
            with st.spinner("Extracting features and predicting..."):
                try:
                    result, y, sr = analyze_audio_bytes(wav_bytes)
                    st.success("Analysis complete.")
                    _show_analysis_result(result, y, sr, audio_bytes=wav_bytes)
                except Exception as e:
                    st.error(f"Failed to analyze the uploaded file: {e}")
        else:
            st.info("Upload a WAV file above to see the analysis results.")

    # ── Tab 3: Live Recording ──────────────────────────────────────────────────
    with tab3:
        st.subheader("Record from Microphone")
        st.caption("The microphone of the machine running Streamlit is used.")

        duration = st.slider("Recording duration (seconds)", min_value=1, max_value=10, value=3, step=1)

        if st.button("Start Recording", type="primary"):
            with st.spinner(f"Recording for {duration} second(s)... Speak now!"):
                try:
                    y, sr = record_audio(duration=float(duration))
                except Exception as e:
                    st.error(f"Microphone error: {e}")
                    st.stop()

            st.success("Recording finished. Analyzing...")

            with st.spinner("Extracting features..."):
                try:
                    result = analyze_audio_array(y, sr)
                except Exception as e:
                    st.error(f"Feature extraction failed: {e}")
                    st.stop()

            # Save recording as WAV bytes for playback
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = Path(f.name)
            try:
                sf.write(str(tmp_path), y, sr)
                audio_bytes = tmp_path.read_bytes()
            finally:
                tmp_path.unlink(missing_ok=True)

            st.success("Analysis complete.")
            _show_analysis_result(result, y, sr, audio_bytes=audio_bytes)

    # ── Tab 4: Model Performance ───────────────────────────────────────────────
    with tab4:
        st.subheader("Model Performance & Optimization Results")

        if opt is None:
            st.info(
                "No optimization report found. "
                "Run the optimization script first:\n\n"
                "```\ncd 'Midterm Project'\npython src/optimize_model.py\n```\n\n"
                "This performs grid search over gender thresholds and three emotion classifiers."
            )
        else:
            # Gender thresholds
            st.markdown("### Gender Classifier (F0 Threshold Optimization)")
            g = opt["gender"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Baseline Accuracy", f"{g['baseline_accuracy']}%")
            col2.metric("Optimized Accuracy", f"{g['optimized_accuracy']}%",
                        delta=f"{g['optimized_accuracy'] - g['baseline_accuracy']:+.2f}%")
            col3.metric(
                "Optimal Thresholds",
                f"Male < {g['optimal_male_female_threshold']} Hz"
                f" | Female < {g['optimal_female_child_threshold']} Hz",
            )

            st.markdown("---")

            # Emotion classifier comparison
            st.markdown("### Emotion Classifier (GridSearchCV, 5-fold CV)")
            e = opt["emotion"]
            col4, col5 = st.columns(2)
            col4.metric("Baseline Accuracy (default GBT)", f"{e['baseline_accuracy']}%")
            col5.metric(
                f"Best Model ({e['best_model']})",
                f"{e['best_cv_accuracy']}%",
                delta=f"{e['best_cv_accuracy'] - e['baseline_accuracy']:+.2f}%",
            )

            model_rows = []
            for name, res in e["models"].items():
                params_str = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in res["best_params"].items())
                model_rows.append({
                    "Model": name,
                    "CV Accuracy (%)": res["cv_accuracy"],
                    "Best Parameters": params_str,
                })
            st.dataframe(
                pd.DataFrame(model_rows).set_index("Model").sort_values("CV Accuracy (%)", ascending=False),
                use_container_width=True,
            )

            st.markdown(f"**Features used:** {', '.join(e['feature_cols'])}")

        st.markdown("---")

        # Confusion matrices side by side
        st.markdown("### Confusion Matrices")
        col_g, col_e = st.columns(2)
        with col_g:
            if not cm.empty:
                st.pyplot(_plot_cm(cm, "Gender Classifier (Rule-based)"))
        with col_e:
            if not ecm.empty:
                st.pyplot(_plot_cm(ecm, "Emotion Classifier (ML)"))

        # Class-level stats bar chart
        if not stats.empty and "Success_%" in stats.columns:
            st.markdown("### Per-Class Success Rate")
            fig, ax = plt.subplots(figsize=(7, 3))
            colors = ["#2196F3", "#E91E63", "#4CAF50"]
            ax.bar(stats["Class"], stats["Success_%"], color=colors[: len(stats)])
            ax.set_ylabel("Success Rate (%)")
            ax.set_ylim(0, 100)
            ax.axhline(y=float(parse_accuracy(SUMMARY_PATH).replace("%", "") or 0),
                       color="red", linestyle="--", linewidth=1, label="Overall accuracy")
            for i, (_, r) in enumerate(stats.iterrows()):
                ax.text(i, r["Success_%"] + 1.5, f"{r['Success_%']:.1f}%", ha="center", fontsize=9)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)


if __name__ == "__main__":
    main()
