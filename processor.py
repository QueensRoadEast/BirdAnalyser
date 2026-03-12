"""
Bird detection pipeline: BirdNET integration, acoustic feature extraction,
spectrogram rendering, and result aggregation.
"""

import io
import os
from dataclasses import dataclass
from typing import Callable, Optional

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer


@dataclass
class DetectionResult:
    file_name: str
    species: str
    confidence: float
    start_time: float
    end_time: float
    duration: float
    category: str
    label: str
    peak_frequency_hz: float
    spectral_centroid_std: float
    noise_floor_rms: float
    spectrogram_png: bytes


ProgressCallback = Optional[Callable[[int, int, Optional[str]], None]]


def merge_detections(
    detections: list[dict], gap_tolerance: float = 0.5
) -> list[dict]:
    """Merge overlapping or near-adjacent detections into contiguous segments."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["start_time"])
    merged = [sorted_dets[0].copy()]

    for det in sorted_dets[1:]:
        prev = merged[-1]
        if det["start_time"] <= prev["end_time"] + gap_tolerance:
            prev["end_time"] = max(prev["end_time"], det["end_time"])
            prev["confidence"] = max(prev["confidence"], det["confidence"])
        else:
            merged.append(det.copy())

    return merged


class BirdProcessor:
    """BirdNET detection + librosa feature-extraction pipeline."""

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(
        self,
        analyzer: Analyzer,
        target_species: str,
        confidence_threshold: float = 0.7,
    ):
        self.analyzer = analyzer
        self.target_species = target_species.strip().lower()
        self.confidence_threshold = confidence_threshold
        self._call_count = 0
        self._song_count = 0

    # ------------------------------------------------------------------
    # Categorisation
    # ------------------------------------------------------------------

    def _categorize(self, duration: float) -> tuple[str, str]:
        if duration < 1.5:
            self._call_count += 1
            return "Call", f"call_{self._call_count}"
        self._song_count += 1
        return "Song", f"song_{self._song_count}"

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _noise_floor_rms(
        y: np.ndarray, sr: int, detection_start: float
    ) -> float:
        """RMS of 0.5 s of audio immediately before the detection."""
        noise_len = int(0.5 * sr)
        start_sample = int(detection_start * sr)
        noise_start = max(0, start_sample - noise_len)
        if noise_start >= start_sample:
            return 0.0
        return float(np.sqrt(np.mean(y[noise_start:start_sample] ** 2)))

    @staticmethod
    def _spectral_features(
        segment: np.ndarray, sr: int
    ) -> tuple[float, float]:
        """Return (peak_frequency_hz, spectral_centroid_std)."""
        magnitudes = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
        peak_freq = float(freqs[np.argmax(magnitudes)])

        centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        centroid_std = float(np.std(centroid))

        return peak_freq, centroid_std

    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------

    @staticmethod
    def _render_spectrogram(
        segment: np.ndarray, sr: int, peak_freq: float, label: str
    ) -> bytes:
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(segment)), ref=np.max
        )
        img = librosa.display.specshow(
            S, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
        )
        ax.axhline(
            y=peak_freq,
            color="cyan",
            linestyle="--",
            linewidth=1.5,
            label=f"Peak Freq: {peak_freq:.0f} Hz",
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(label, fontsize=12)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------

    def process_file(self, file_path: str) -> list[DetectionResult]:
        recording = Recording(
            self.analyzer,
            file_path,
            min_conf=self.confidence_threshold,
        )
        recording.analyze()

        matched = [
            d
            for d in recording.detections
            if self.target_species in d["common_name"].lower()
        ]
        merged = merge_detections(matched)
        if not merged:
            return []

        y, sr = librosa.load(file_path, sr=None)
        file_name = os.path.basename(file_path)
        results: list[DetectionResult] = []

        for det in merged:
            start, end = det["start_time"], det["end_time"]
            duration = end - start
            category, label = self._categorize(duration)

            seg_start = int(start * sr)
            seg_end = min(int(end * sr), len(y))
            segment = y[seg_start:seg_end]
            if len(segment) == 0:
                continue

            peak_freq, centroid_std = self._spectral_features(segment, sr)
            noise_rms = self._noise_floor_rms(y, sr, start)
            spectrogram = self._render_spectrogram(
                segment, sr, peak_freq, label
            )

            results.append(
                DetectionResult(
                    file_name=file_name,
                    species=det["common_name"],
                    confidence=round(det["confidence"], 4),
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    duration=round(duration, 3),
                    category=category,
                    label=label,
                    peak_frequency_hz=round(peak_freq, 2),
                    spectral_centroid_std=round(centroid_std, 2),
                    noise_floor_rms=round(noise_rms, 6),
                    spectrogram_png=spectrogram,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_files(
        self,
        file_paths: list[str],
        progress_callback: ProgressCallback = None,
    ) -> list[DetectionResult]:
        all_results: list[DetectionResult] = []

        for idx, path in enumerate(file_paths):
            try:
                all_results.extend(self.process_file(path))
            except Exception as exc:
                if progress_callback:
                    progress_callback(idx + 1, len(file_paths), str(exc))
                continue
            if progress_callback:
                progress_callback(idx + 1, len(file_paths), None)

        return all_results
