"""
Bird detection pipeline: BirdNET integration, multi-peak acoustic feature
extraction, MFCC-based spectral clustering, noise profiling, spectrogram
rendering, and result aggregation.
"""

import io
import os
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

MAX_PEAKS = 5
NOISE_WINDOW_S = 0.5
N_MFCC = 13


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class FrequencyPeak:
    frequency_hz: float
    magnitude_db: float
    snr_db: float
    spectral_spread_hz: float
    is_dominant: bool = False


@dataclass
class NoiseProfile:
    rms_amplitude: float
    avg_frequency_hz: float
    freq_range_low_hz: float
    freq_range_high_hz: float
    avg_magnitude_db: float


@dataclass
class DetectionResult:
    file_name: str
    species: str
    confidence: float
    start_time: float
    end_time: float
    duration: float
    duration_type: str
    spectral_type: str = ""
    spectral_label: str = ""
    peaks: list[FrequencyPeak] = field(default_factory=list)
    spectral_centroid_std: float = 0.0
    noise: NoiseProfile = field(
        default_factory=lambda: NoiseProfile(0, 0, 0, 0, 0)
    )
    spectrogram_png: bytes = b""
    fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(N_MFCC))


ProgressCallback = Optional[Callable[[int, int, Optional[str]], None]]


# ======================================================================
# Helpers
# ======================================================================

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


def _safe_db(amplitude: float) -> float:
    """Convert linear amplitude to dB, clamping at -120 dB."""
    return float(20.0 * np.log10(max(amplitude, 1e-6)))


def _cluster_id_to_letter(cluster_id: int) -> str:
    """Map 0 -> 'A', 1 -> 'B', ... 25 -> 'Z', 26 -> 'AA', etc."""
    letters = string.ascii_uppercase
    result = ""
    n = cluster_id
    while True:
        result = letters[n % 26] + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


# ======================================================================
# Spectral clustering (Pass 2)
# ======================================================================

def classify_detections(
    results: list[DetectionResult],
    distance_threshold: float = 0.3,
) -> list[DetectionResult]:
    """Cluster detections by MFCC fingerprint similarity and assign labels."""
    if not results:
        return results

    if len(results) == 1:
        results[0].spectral_type = "type_A"
        results[0].spectral_label = "type_A_1"
        return results

    matrix = np.stack([r.fingerprint for r in results])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    matrix_normed = matrix / norms

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(matrix_normed)

    instance_counters: Counter[int] = Counter()
    for r, cluster_id in zip(results, labels):
        letter = _cluster_id_to_letter(int(cluster_id))
        r.spectral_type = f"type_{letter}"
        instance_counters[cluster_id] += 1
        r.spectral_label = f"type_{letter}_{instance_counters[cluster_id]}"

    return results


# ======================================================================
# Processor
# ======================================================================

class BirdProcessor:
    """BirdNET detection + multi-peak librosa feature-extraction pipeline."""

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

    # ------------------------------------------------------------------
    # Duration-based type (secondary label)
    # ------------------------------------------------------------------

    @staticmethod
    def _duration_type(duration: float) -> str:
        return "Call" if duration < 1.5 else "Song"

    # ------------------------------------------------------------------
    # Spectral fingerprint (MFCC)
    # ------------------------------------------------------------------

    @staticmethod
    def _spectral_fingerprint(segment: np.ndarray, sr: int) -> np.ndarray:
        """Return a fixed-length MFCC mean vector as the spectral fingerprint."""
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
        return np.mean(mfccs, axis=1)

    # ------------------------------------------------------------------
    # Noise profiling
    # ------------------------------------------------------------------

    @staticmethod
    def _build_noise_profile(
        y: np.ndarray, sr: int, detection_start: float, detection_end: float
    ) -> NoiseProfile:
        noise_len = int(NOISE_WINDOW_S * sr)
        start_sample = int(detection_start * sr)
        end_sample = int(detection_end * sr)
        noise_start = max(0, start_sample - noise_len)

        if noise_start < start_sample and start_sample > 0:
            noise_seg = y[noise_start:start_sample]
        elif end_sample < len(y):
            fallback_end = min(len(y), end_sample + noise_len)
            noise_seg = y[end_sample:fallback_end]
        else:
            return NoiseProfile(0.0, 0.0, 0.0, 0.0, -120.0)

        if len(noise_seg) == 0:
            return NoiseProfile(0.0, 0.0, 0.0, 0.0, -120.0)
        rms = float(np.sqrt(np.mean(noise_seg**2)))

        magnitudes = np.abs(np.fft.rfft(noise_seg))
        freqs = np.fft.rfftfreq(len(noise_seg), d=1.0 / sr)

        mag_sum = magnitudes.sum()
        if mag_sum < 1e-12:
            return NoiseProfile(rms, 0.0, 0.0, 0.0, _safe_db(rms))

        weights = magnitudes / mag_sum
        avg_freq = float(np.dot(freqs, weights))

        cumulative = np.cumsum(magnitudes)
        total = cumulative[-1]
        low_idx = int(np.searchsorted(cumulative, total * 0.1))
        high_idx = int(np.searchsorted(cumulative, total * 0.9))
        low_idx = np.clip(low_idx, 0, len(freqs) - 1)
        high_idx = np.clip(high_idx, 0, len(freqs) - 1)

        return NoiseProfile(
            rms_amplitude=round(rms, 8),
            avg_frequency_hz=round(avg_freq, 1),
            freq_range_low_hz=round(float(freqs[low_idx]), 1),
            freq_range_high_hz=round(float(freqs[high_idx]), 1),
            avg_magnitude_db=round(_safe_db(rms), 2),
        )

    # ------------------------------------------------------------------
    # Multi-peak extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _find_peaks(
        segment: np.ndarray, sr: int, noise_rms: float
    ) -> list[FrequencyPeak]:
        magnitudes = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
        mag_db = 20.0 * np.log10(np.maximum(magnitudes, 1e-12))

        kernel_size = max(3, len(magnitudes) // 200) | 1
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(mag_db, kernel, mode="same")

        indices, _ = find_peaks(
            smoothed,
            prominence=6.0,
            distance=max(1, int(50 * len(segment) / sr)),
        )

        if len(indices) == 0:
            peak_idx = int(np.argmax(mag_db))
            indices = np.array([peak_idx])

        sorted_order = np.argsort(-mag_db[indices])
        top_indices = indices[sorted_order[:MAX_PEAKS]]

        noise_db = _safe_db(noise_rms) if noise_rms > 0 else -120.0
        dominant_idx = top_indices[0]

        peaks: list[FrequencyPeak] = []
        for idx in top_indices:
            freq = float(freqs[idx])
            m_db = float(mag_db[idx])
            snr = m_db - noise_db

            bw_low = max(0, idx - max(1, len(magnitudes) // 100))
            bw_high = min(len(magnitudes), idx + max(1, len(magnitudes) // 100))
            local_freqs = freqs[bw_low:bw_high]
            local_mags = magnitudes[bw_low:bw_high]
            local_sum = local_mags.sum()
            if local_sum > 1e-12:
                local_weights = local_mags / local_sum
                local_mean = np.dot(local_freqs, local_weights)
                spread = float(
                    np.sqrt(np.dot((local_freqs - local_mean) ** 2, local_weights))
                )
            else:
                spread = 0.0

            peaks.append(
                FrequencyPeak(
                    frequency_hz=round(freq, 1),
                    magnitude_db=round(m_db, 2),
                    snr_db=round(snr, 2),
                    spectral_spread_hz=round(spread, 1),
                    is_dominant=(idx == dominant_idx),
                )
            )

        return peaks

    @staticmethod
    def _spectral_centroid_std(segment: np.ndarray, sr: int) -> float:
        centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        return float(np.std(centroid))

    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------

    @staticmethod
    def _render_spectrogram(
        segment: np.ndarray,
        sr: int,
        peaks: list[FrequencyPeak],
        noise: NoiseProfile,
        result: DetectionResult,
    ) -> bytes:
        fig, ax = plt.subplots(figsize=(12, 5))
        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(segment)), ref=np.max
        )
        img = librosa.display.specshow(
            S, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
        )

        if noise.freq_range_high_hz > noise.freq_range_low_hz:
            ax.axhspan(
                noise.freq_range_low_hz,
                noise.freq_range_high_hz,
                color="white",
                alpha=0.08,
            )
            ax.axhline(
                y=noise.avg_frequency_hz,
                color="white",
                linestyle=":",
                linewidth=1.0,
                alpha=0.5,
            )

        peak_colors = plt.cm.cool(np.linspace(0.2, 0.9, max(len(peaks), 1)))

        for i, pk in enumerate(peaks):
            if pk.is_dominant:
                ax.axhline(
                    y=pk.frequency_hz,
                    color="cyan",
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.95,
                )
                ax.text(
                    0.01,
                    pk.frequency_hz,
                    f" {pk.frequency_hz:.0f} Hz  SNR {pk.snr_db:+.1f} dB  [DOMINANT]",
                    color="cyan",
                    fontsize=8,
                    fontweight="bold",
                    va="bottom",
                    transform=ax.get_yaxis_transform(),
                    bbox=dict(
                        facecolor="black", alpha=0.6, pad=1.5, edgecolor="none"
                    ),
                )
            else:
                ax.axhline(
                    y=pk.frequency_hz,
                    color=peak_colors[i],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.7,
                )
                ax.text(
                    0.01,
                    pk.frequency_hz,
                    f" {pk.frequency_hz:.0f} Hz  SNR {pk.snr_db:+.1f} dB",
                    color=peak_colors[i],
                    fontsize=7,
                    va="bottom",
                    transform=ax.get_yaxis_transform(),
                    bbox=dict(
                        facecolor="black", alpha=0.5, pad=1, edgecolor="none"
                    ),
                )

        noise_text = (
            f"Noise Floor\n"
            f"  RMS: {noise.rms_amplitude:.6f}\n"
            f"  Avg: {noise.avg_frequency_hz:.0f} Hz\n"
            f"  Range: {noise.freq_range_low_hz:.0f}\u2013{noise.freq_range_high_hz:.0f} Hz\n"
            f"  Level: {noise.avg_magnitude_db:.1f} dB"
        )
        ax.text(
            0.99,
            0.02,
            noise_text,
            transform=ax.transAxes,
            fontsize=7,
            color="white",
            va="bottom",
            ha="right",
            family="monospace",
            bbox=dict(
                facecolor="black", alpha=0.7, pad=4, edgecolor="gray", linewidth=0.5
            ),
        )

        legend_handles = [
            mpatches.Patch(color="cyan", label="Dominant Peak"),
            mpatches.Patch(color="gray", label="Secondary Peaks"),
            mpatches.Patch(facecolor="white", alpha=0.3, label="Noise Freq Range"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=7,
            framealpha=0.7,
        )

        title = f"{result.spectral_label}  [{result.duration_type}]"
        ax.set_title(title, fontsize=12)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Per-file processing (Pass 1)
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

            seg_start = int(start * sr)
            seg_end = min(int(end * sr), len(y))
            segment = y[seg_start:seg_end]
            if len(segment) == 0:
                continue

            noise = self._build_noise_profile(y, sr, start, end)
            peaks = self._find_peaks(segment, sr, noise.rms_amplitude)
            centroid_std = self._spectral_centroid_std(segment, sr)
            fp = self._spectral_fingerprint(segment, sr)

            results.append(
                DetectionResult(
                    file_name=file_name,
                    species=det["common_name"],
                    confidence=round(det["confidence"], 4),
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    duration=round(duration, 3),
                    duration_type=self._duration_type(duration),
                    peaks=peaks,
                    spectral_centroid_std=round(centroid_std, 2),
                    noise=noise,
                    fingerprint=fp,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Batch processing (Pass 1 + Pass 2)
    # ------------------------------------------------------------------

    def process_files(
        self,
        file_paths: list[str],
        progress_callback: ProgressCallback = None,
        similarity_threshold: float = 0.3,
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

        classify_detections(all_results, similarity_threshold)

        for r in all_results:
            seg_start = None
            seg_end = None
            # Re-derive segment for spectrogram rendering.
            # To avoid re-loading audio, we stored fingerprint; now we need
            # the segment for the spectrogram. We cache audio per file.
            pass

        self._render_all_spectrograms(all_results, file_paths)

        return all_results

    def _render_all_spectrograms(
        self,
        results: list[DetectionResult],
        file_paths: list[str],
    ) -> None:
        """Render spectrograms after clustering labels are assigned."""
        audio_cache: dict[str, tuple[np.ndarray, int]] = {}

        for r in results:
            fpath = None
            for p in file_paths:
                if os.path.basename(p) == r.file_name:
                    fpath = p
                    break
            if fpath is None:
                continue

            if fpath not in audio_cache:
                audio_cache[fpath] = librosa.load(fpath, sr=None)

            y, sr = audio_cache[fpath]
            seg_start = int(r.start_time * sr)
            seg_end = min(int(r.end_time * sr), len(y))
            segment = y[seg_start:seg_end]
            if len(segment) == 0:
                continue

            r.spectrogram_png = self._render_spectrogram(
                segment, sr, r.peaks, r.noise, r
            )
