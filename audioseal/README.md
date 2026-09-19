# MeanVC2 + AudioSeal: Real-Time Streaming Watermarking & Interactive Web Studio

This module provides low-latency, strictly causal, continuous **AudioSeal neural audio watermarking** (embedding & real-time extraction) integrated with **MeanVC2** (streaming zero-shot voice conversion), complete with a full-duplex Web Audio studio and an offline evaluation benchmark suite.

---

## Directory Structure

```text
MeanVC2/audioseal/
├── __init__.py                # Module export interface
├── watermarker.py             # AudioSealStreamWatermarker core streaming watermarker class
├── vc_watermark_runner.py     # MeanVC2 streaming inference + watermarking wrapper (WatermarkedVCRunner)
├── server_realtime_web.py     # Full-duplex WebSocket + aiohttp real-time streaming server
├── run_web_studio.sh          # One-click launcher script with port conflict auto-recovery
├── run_web_studio_gpu.pjm     # Genkai Supercomputer GPU batch submission script
├── eval_meanvc_watermark.py   # Large-scale offline benchmark (BER, SNR, PESQ, STOI)
├── eval_meanvc_watermark.pjm  # Benchmark evaluation job script for H100 GPUs
├── eval_attacks.py            # Robustness evaluation attack suite (noise, resample, MP3, etc.)
├── web/                       # Zero-dependency glassmorphism dark-mode UI
│   ├── index.html             # Web Studio layout structure
│   ├── style.css              # Modern responsive styling
│   └── app.js                 # Web Audio API 16kHz capture, oscilloscope & anti-lag playback queue
├── exp/                       # Benchmark experiment outputs and summary reports
└── logs/                      # Job execution and server logs
```

---

## Key Features

1. **Strictly Causal Real-Time Streaming (Zero Lookahead)**:
   - Built on `audioseal_wm_streaming` causal convolutional architecture with zero future lookahead.
   - Operates on streaming chunks on the fly (40ms, 120ms, or 160ms steps) with negligible overhead ($\text{RTF} < 0.05$).
2. **Full-Stack Anti-Lag & Zero Drift**:
   - **Client-Side Catch-Up**: The Web Audio player dynamically monitors queue backlog; drifts over 280ms instantly snap back to real time.
   - **Leaky Server Queue**: An asynchronous FIFO queue (max size = 2) drops stale backlog during network or CPU hiccups to process the latest speech.
   - **VAD Pause Attention Reset**: Automatically refreshes the DiT KV-cache during natural conversational pauses to prevent attention sequence length from growing and slowing down inference.
3. **High Density Redundancy & 100% Blind Extraction**:
   - The 16-bit secret payload is broadcast across every temporal frame.
   - Slices as short as 60ms are sufficient to reliably recover the 16-bit signature.
4. **Zero-Latency Timbre Hot-Switching**:
   - Precomputes speaker embeddings and GTM KV caches at startup, enabling instantaneous ($0\,\text{ms}$) switching between target timbres (warm/clear female, deep/default male) without restarting or reloading models.

---

## Quick Start

### 1. Launch Real-Time Web Studio (Recommended)

Run the studio launcher on your server or local machine:
```bash
cd audioseal
./run_web_studio.sh 8998 cuda 40ms
```

If running on a remote server, forward port `8998` over SSH to your local machine:
```bash
ssh -L 8998:localhost:8998 <user>@<server_ip>
```

Open Chrome or Edge and navigate to:
```text
http://localhost:8998
```
Select your desired target speaker timbre (e.g. `LibriTTS Female (Spk 150 · Warm)`), click **Start Microphone**, and speak to hear real-time converted and watermarked audio along with live 16-bit detection metrics.

### 2. End-to-End Offline File Processing

```python
from audioseal.vc_watermark_runner import WatermarkedVCRunner

# Initialize combined VC + Watermarking runner
runner = WatermarkedVCRunner(
    target_wav="test_audio/common_voice_en_10119832.wav",
    device="cpu",
    model="40ms",
    alpha=0.5,
    secret_message=[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
)

# Convert audio file and verify watermark detection accuracy
runner.process_file("input.wav", "output_watermarked.wav", verify_watermark=True)
```

### 3. Run Large-Scale Benchmark Evaluation

Run the batch evaluation script on LibriTTS test sets:
```bash
pjsub eval_meanvc_watermark.pjm
```
Evaluates audio quality (SNR, PESQ, STOI), raw detection accuracy, and Bit Error Rate (BER) across diverse attack conditions (Gaussian noise, bandpass filtering, lossy MP3 compression, and tempo perturbation).
