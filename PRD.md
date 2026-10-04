# PRD.md

## 1. Project Overview
**Project Name:** EdgeWake
**Target Objective:** ISRO SIH Problem Statement 26172
**Mission:** Minimize wake-to-ASR handoff latency for space and edge telemetry systems while strictly adhering to severe edge resource constraints (<256 KB RAM, <10% idle CPU).

## 2. Core Problem
Traditional edge voice assistants use a sequential pipeline: `Wake -> Connect Wi-Fi -> Open Socket -> Stream Audio`. This introduces unacceptable latency (1.5s+) and often clips the beginning of the user's command. Furthermore, monolithic architectures create single points of failure if cloud connectivity is denied.

## 3. Product Innovations (The EdgeWake Solution)
*   **Persistent Wake-to-ASR Handoff:** Network connections are established at boot and maintained via heartbeat.
*   **Pre-Roll Preservation:** A static 16 KB circular buffer continuously stores the last 500 ms of audio, ensuring instantaneous, context-complete streaming the moment the wake word is confirmed.
*   **Adaptive Acoustic Gating:** A lightweight Exponential Moving Average (EMA) Voice Activity Detector (VAD) prevents expensive feature extraction and neural inference during acoustic inactivity.
*   **Algorithmic Diversity:** A parallel, offline Dynamic Time Warping (DTW) matcher acts as an emergency fallback, triggering local hardware actuation if network or primary AI fails.

## 4. Empirical Validation Targets (Success Metrics)
| Metric | PS-26172 Requirement | EdgeWake Validation Target |
| :--- | :--- | :--- |
| **Peak RAM** | < 256 KB | Measured: [Pending] KB |
| **Idle CPU** | < 10% | Measured: [Pending] % |
| **Wake-to-ASR Latency** | Minimize | Measured: [Pending] ms |
| **True Positive Rate (TPR)** | High | Measured: [Pending] % |
| **False Activations** | Near-zero | Measured: [Pending] / hour |
| **Data Overhead** | Minimal | Measured: [Pending] KB/s |

## 5. Hardware & Ecosystem
*   **Reference Hardware:** ESP-WROOM-32 (Xtensa LX6)
*   **Sensor:** INMP441 I2S Digital MEMS Microphone
*   **Cloud Endpoint:** Node.js / Python WebSocket Server + Streaming ASR (Faster-Whisper/Vosk)

## 6. Submission Compliance
*   **Open-source only** — TFLM + ESP-IDF + custom DTW; zero proprietary/closed SDKs.
*   **Custom keyword** — "Pinaka" is trained by the team; no pre-trained global keywords (no "Hey Google"/"Alexa").
*   **No heavy transformers on-device** — transformer-based ASR runs in the cloud; the edge runs a tiny int8 KWS model only.
*   Full criteria-by-criteria audit: see [DISQUALIFIER-AUDIT.md](DISQUALIFIER-AUDIT.md).
*   Measured values live in [MEASUREMENTS.md](MEASUREMENTS.md) — nothing here is a claim.