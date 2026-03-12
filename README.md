# BirdAnalyser

Automated bird species detection and acoustic analysis powered by [BirdNET](https://github.com/kahst/BirdNET-Analyzer). Upload audio recordings, specify a target species, and get detailed acoustic feature extraction with downloadable results.

## Features

- **BirdNET Detection** — Identify bird species in audio files with configurable confidence thresholds.
- **Smart Merging** — Overlapping or adjacent detections are merged into contiguous segments.
- **Call / Song Categorisation** — Detections are labelled as *Call* (< 1.5 s) or *Song* (>= 1.5 s) with auto-incrementing IDs.
- **Acoustic Feature Extraction** (via librosa):
  - Background noise floor (RMS of the 0.5 s preceding each detection)
  - Peak frequency (Hz) from FFT
  - Spectral centroid standard deviation
- **Spectrogram Visualisation** — Per-detection spectrograms with the peak frequency highlighted.
- **CSV Export** — One-click download of all results.

## Project Structure

```
BirdAnalyser/
├── app.py              # Streamlit web interface
├── processor.py        # Detection pipeline & feature extraction
├── requirements.txt    # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your `PATH` (required by librosa / birdnetlib)

## Installation

```bash
# Clone or download the project, then:
cd BirdAnalyser
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

1. Open the URL shown in your terminal (default `http://localhost:8501`).
2. In the sidebar, enter the **Target Species** common name (e.g. *House Sparrow*).
3. Adjust the **Confidence Threshold** slider (default 0.70).
4. Upload one or more audio files (WAV, MP3, FLAC, OGG, M4A).
5. Click **Run Analysis**.
6. Browse the results table and spectrograms, then download the CSV.

## Technical Details

| Component | Library |
|---|---|
| AI detection engine | `birdnetlib` (BirdNET-Analyzer wrapper) |
| Signal processing | `librosa`, `numpy`, `scipy` |
| Visualisation | `matplotlib` |
| Web interface | `streamlit` |
| Data export | `pandas` |

### How It Works

1. **Detection** — Each uploaded file is analysed by BirdNET. Only detections matching the target species name and exceeding the confidence threshold are kept.
2. **Merging** — Consecutive detections within 0.5 s of each other are merged into a single segment to capture full vocalisations.
3. **Categorisation** — Merged segments shorter than 1.5 s are labelled *Call*; longer segments are labelled *Song*. Labels auto-increment per session (`call_1`, `call_2`, `song_1`, ...).
4. **Feature Extraction** — For each segment, the processor computes the noise floor RMS, FFT peak frequency, and spectral centroid standard deviation.
5. **Visualisation** — A spectrogram is rendered for every detection with the peak frequency overlaid as a dashed cyan line.
6. **Export** — All numeric results are compiled into a downloadable CSV.

## License

This project is provided as-is for educational and research purposes.
