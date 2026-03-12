"""
Streamlit front-end for BirdAnalyser.

Run with:  streamlit run app.py
"""

import os
import tempfile

import pandas as pd
import streamlit as st
from birdnetlib.analyzer import Analyzer

from processor import BirdProcessor, DetectionResult


@st.cache_resource(show_spinner="Loading BirdNET model...")
def _load_analyzer() -> Analyzer:
    return Analyzer()


def _results_to_dataframe(results: list[DetectionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        dominant = next((p for p in r.peaks if p.is_dominant), None)
        rows.append(
            {
                "File": r.file_name,
                "Species": r.species,
                "Confidence": r.confidence,
                "Start (s)": r.start_time,
                "End (s)": r.end_time,
                "Duration (s)": r.duration,
                "Spectral Type": r.spectral_type,
                "Label": r.spectral_label,
                "Duration Type": r.duration_type,
                "Dominant Peak (Hz)": dominant.frequency_hz if dominant else 0,
                "Dominant SNR (dB)": dominant.snr_db if dominant else 0,
                "Num Peaks": len(r.peaks),
                "Centroid StdDev": r.spectral_centroid_std,
                "Noise RMS": r.noise.rms_amplitude,
                "Noise Avg Freq (Hz)": r.noise.avg_frequency_hz,
                "Noise Range (Hz)": (
                    f"{r.noise.freq_range_low_hz:.0f}\u2013{r.noise.freq_range_high_hz:.0f}"
                ),
                "Noise Level (dB)": r.noise.avg_magnitude_db,
            }
        )
    return pd.DataFrame(rows)


def _show_results(results: list[DetectionResult]) -> None:
    if not results:
        st.warning(
            "No detections matched the specified species and confidence threshold."
        )
        return

    st.success(f"**{len(results)}** detection(s) found.")

    types = sorted({r.spectral_type for r in results})
    type_summary = ", ".join(
        f"**{t}** ({sum(1 for r in results if r.spectral_type == t)})"
        for t in types
    )
    st.info(f"Spectral groups: {type_summary}")

    df = _results_to_dataframe(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="bird_analysis_results.csv",
        mime="text/csv",
    )

    st.subheader("Spectrograms")
    cols = st.columns(2)
    for idx, r in enumerate(results):
        with cols[idx % 2]:
            caption = (
                f"{r.spectral_label} [{r.duration_type}]"
                f"  \u2014  {r.species}  ({r.file_name})"
            )
            st.image(
                r.spectrogram_png,
                caption=caption,
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(page_title="BirdAnalyser", layout="wide")
    st.title("BirdAnalyser")
    st.caption(
        "Automated bird species detection and acoustic analysis powered by BirdNET"
    )

    # -- Sidebar -------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")

        target_species = st.text_input(
            "Target Species",
            placeholder="e.g. House Sparrow",
            help="Common English name of the bird species to detect.",
        )

        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.70,
            step=0.05,
            help="Minimum BirdNET confidence to keep a detection.",
        )

        similarity = st.slider(
            "Similarity Threshold",
            min_value=0.05,
            max_value=1.00,
            value=0.30,
            step=0.05,
            help=(
                "Cosine distance threshold for spectral grouping. "
                "Lower = stricter (more groups), higher = looser (fewer groups)."
            ),
        )

        st.divider()

        uploaded_files = st.file_uploader(
            "Upload Audio Files",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            accept_multiple_files=True,
        )

        can_run = bool(uploaded_files) and bool(target_species.strip())
        run_clicked = st.button(
            "Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )

    # -- Main area -----------------------------------------------------
    if "results" not in st.session_state:
        st.session_state.results = None

    if run_clicked:
        analyzer = _load_analyzer()
        processor = BirdProcessor(analyzer, target_species, confidence)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[str] = []
            for f in uploaded_files:
                dest = os.path.join(tmpdir, f.name)
                with open(dest, "wb") as fp:
                    fp.write(f.getbuffer())
                paths.append(dest)

            bar = st.progress(0, text="Analysing...")
            errors: list[str] = []

            def _on_progress(
                done: int, total: int, error: str | None = None
            ) -> None:
                bar.progress(
                    done / total,
                    text=f"Processing file {done}/{total}",
                )
                if error:
                    errors.append(error)

            results = processor.process_files(
                paths, _on_progress, similarity_threshold=similarity
            )
            bar.empty()

        if errors:
            with st.expander(f"{len(errors)} file(s) had errors"):
                for e in errors:
                    st.error(e)

        st.session_state.results = results

    if st.session_state.results is None:
        st.info(
            "Upload audio files and set the target species in the sidebar, "
            "then press **Run Analysis**."
        )
    else:
        _show_results(st.session_state.results)


if __name__ == "__main__":
    main()
