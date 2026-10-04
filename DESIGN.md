# DESIGN.md

## 1. Acoustic Configurations
*   **Pre-Roll Buffer:** 500 ms (8,000 samples, 16 KB `int16_t`).
*   **VAD Window:** 20 ms frames (320 samples).
*   **VAD EMA Alpha:** 0.05 baseline (slow adaptation to background noise). If measured onset latency > 100 ms, tune toward 0.12–0.2 and re-verify [MEASUREMENTS.md](MEASUREMENTS.md).
*   **Feature Extraction Window:** 30 ms (480 samples) with 20 ms stride → ~50 Hz feature rate.
*   **MFCC:** 40 mel filters, log-mel, DCT-II, float precision (parity with PC `.wav` testing).
*   **End of Speech (EOS) Timeout:** Configurable default of 800 ms.

**Known risk:** Xtensa LX6 has no FPU. Float MFCC frame time is the #1 threat to the <10% CPU budget; measured in [MEASUREMENTS.md](MEASUREMENTS.md) with fallbacks (fixed-point MFCC, 26–32 mel filters, ESP-DSP FFT).

## 2. Telemetry Dashboard (SIH Pitch UI)
To prove the system's low latency and resource efficiency during the live demonstration, a local Node.js/React dashboard will mirror the ESP32's state in real-time.

**Visual Layout:**
*   **Header:** "EdgeWake Live Telemetry - ESP-WROOM-32"
*   **Timeline Graph (The Core USP):**
    *   Mic Input -> [Wake Word Detected] -> Pre-Roll TX -> ASR Response.
    *   Dynamically updates with the measured `T3 - T0` latency in milliseconds.
*   **System Diagnostics Panel:**
    *   Free Heap: `[Live Value] KB`
    *   Idle CPU: `[Live Value] %`
    *   Feature Extraction Time: `[Live Value] ms`
    *   KWS Inference Time: `[Live Value] ms`

## 3. Hardware Interfacing (Status LEDs)
*   **Blue LED (Solid):** Persistent WebSocket Connected.
*   **Green LED (Blink):** Active audio streaming to cloud.
*   **Red LED (Solid):** Local DTW Emergency Abort triggered (Cloud bypassed).
