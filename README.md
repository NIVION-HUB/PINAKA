# Project PINAKA: Ultra-Lightweight Edge Voice Activator

**Smart India Hackathon (SIH) 2026**
**Problem Statement:** Low Latency and Efficient Voice Activator for Edge Devices

---

## 🚀 The Vision (Aerospace Context)
Traditional voice-controlled IoT devices process everything in the cloud, assuming high-bandwidth Wi-Fi and perfect acoustics. For ISRO aerospace missions and deep-space telemetry, this paradigm completely fails. 

**Project PINAKA** solves this by implementing a true **Hybrid Edge-to-Cloud architecture engineered for space**. The heavy lifting of "always-on listening" is done entirely offline on a $5 microcontroller using a **36 KB Neural Network**. The edge device stays asleep, waking up only when the exact custom keyword "PINAKA" is spoken. Once verified, it securely and efficiently hands off the subsequent audio command to a remote Automated Speech Recognition (ASR) server for complex natural language processing.

---

## 🏆 Key Performance Metrics (SIH Evaluation Constraints)

| Metric | Required Constraint | PINAKA Performance |
|--------|---------------------|--------------------|
| **Model Footprint** | < 256 KB RAM | **36 KB** (INT8 Quantized DS-CNN) |
| **Idle CPU Usage** | < 10% | **~0%** (VAD sleep gating) |
| **Edge Hardware** | Low-power MCU | **ESP32** (Dual-Core Xtensa LX6) |
| **Wake-to-Stream Latency**| "Minimal" | **< 300 ms** (Measured via perf_counter) |
| **Pre-trained Models** | NONE | **Built entirely from scratch** (Custom Wake Word) |

---

## ⚡ Core Innovations (Why PINAKA is Different)

While most teams simply run a Python script and stream raw audio, Project PINAKA implements enterprise-grade, hardware-aware edge engineering:

### 1. 🔋 The 3-Layer Battery Shield
Before the neural network is allowed to consume battery, the audio must pass three hardware/math gates:
1.  **Hardware DMA Transfer:** Audio is captured using I2S Direct Memory Access. Zero CPU cycles are wasted on capturing sound.
2.  **Adaptive Volume Gate:** On boot, the device measures the ambient room noise for 2 seconds. The AI sleeps until volume spikes above this dynamic baseline.
3.  **Human PSD Pre-Filter:** A microsecond Fast Fourier Transform (FFT) checks if the acoustic energy is between **300Hz–3400Hz**. Dog barks and car horns are dropped instantly before the AI ever wakes up.

### 2. 🔐 Opus Compression & AES Encryption
The problem statement strictly demands *minimal data overhead* and mentions *privacy-invasive clouds*.
*   **Compression:** We do not stream raw 256 kbps PCM audio. PINAKA compresses the voice using the **Opus codec (8 kbps)**, reducing Wi-Fi bandwidth by **96%**.
*   **Encryption:** The stream is encrypted using the ESP32's hardware **AES-128 crypto engine**, ensuring the audio cannot be intercepted on public Wi-Fi networks.

### 3. ⏱️ Multi-Scale Temporal Verification
To achieve near-zero false positives, PINAKA doesn't just check the audio once.
*   **Multi-Scale:** It checks 3 different time windows (0.75s, 1.0s, 1.25s) simultaneously to catch incredibly fast or extremely slow speakers.
*   **Temporal Consistency:** It mathematically requires the AI to hit >95% confidence on multiple consecutive 100ms scans before confirming a trigger. A random noise might spike to 95% once, but it cannot hold it.

### 4. 🛰️ Mission-Specific Acoustic Modeling
The AI was not trained on generic street noise. To guarantee performance in aerospace environments, the dataset was augmented to simulate three distinct mission phases:
*   **Earth (Ground Station):** High-frequency human babble and room echo.
*   **Launch (Mid-Journey):** Extreme low-frequency mechanical rumble and pitch-shifted vocal strain (simulating G-forces).
*   **Space (Orbit):** Low-pass filters simulating muffled speech through polycarbonate spacesuit visors.

### 5. 🔲 ESP32 Dual-Core Task Pinning
The ESP32 has two CPU cores. 
*   **Core 1** is pinned 100% to real-time audio inference (it never stops listening).
*   **Core 0** handles the Wi-Fi connection and Cloud ASR WebSocket streaming.
By dividing the tasks, the wake-to-stream latency drops to almost zero because the network connection opens in parallel to the audio capture.

---

## 🧠 Model Architecture & Dataset

*   **Network Type:** Depthwise Separable Convolutional Neural Network (DS-CNN)
*   **Features:** 49x40 Log-Mel Spectrograms
*   **Precision:** INT8 Quantized (via TensorFlow Lite for Microcontrollers)
*   **Training Dataset:** 24,700 samples (Synthesized positive voices mixed with ESC-50 environmental background noise, and 10,000 negative human speech commands).

---

## 🛠️ Repository Structure

*   `/firmware` - The C++ ESP-IDF / Arduino code containing the dual-core runtime and I2S microphone drivers.
*   `/training` - The complete Python pipeline (Dataset Generation, Spectrogram Feature Extraction, Keras Model Training, and TFLite Quantization).
*   `virtual_test.py` - The desktop simulator for testing the `.keras` model with real-time latency measurements before flashing to hardware.
*   `record_my_voice.py` - A utility to inject 20 samples of a specific user's voice into the dataset for extreme personalized accuracy.

---

*Built for Smart India Hackathon 2026. Fully Open Source.*
