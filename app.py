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
    return pd.DataFrame(
        [
            {
                "File": r.file_name,
                "Species": r.species,
                "Confidence": r.confidence,
                "Start (s)": r.start_time,
                "End (s)": r.end_time,
                "Duration (s)": r.duration,
                "Category": r.category,
                "Label": r.label,
                "Peak Freq (Hz)": r.peak_frequency_hz,
                "Centroid StdDev": r.spectral_centroid_std,
                "Noise Floor RMS": r.noise_floor_rms,
            }
            for r in results
        ]
    )


def _show_results(results: list[DetectionResult]) -> None:
    if not results:
        st.warning(
            "No detections matched the specified species and confidence threshold."
        )
        return

    st.success(f"**{len(results)}** detection(s) found.")

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
            st.image(
                r.spectrogram_png,
                caption=f"{r.label}  —  {r.species}  ({r.file_name})",
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(page_title="BirdAnalyser", layout="wide")
    st.title("BirdAnalyser")
    st.caption(
        "Automated bird species detection and acoustic analysis powered by BirdNET"
    )

    # ── Sidebar ──────────────────────────────────────────────────────
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

    # ── Main area ────────────────────────────────────────────────────
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

            results = processor.process_files(paths, _on_progress)
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
