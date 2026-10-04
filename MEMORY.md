# MEMORY.md — Context & Key Decisions Ledger

## Current State
Architecture frozen via multi-round LLM council + official SIH-26172 disqualifier audit. Moving into P0 implementation (Hardware baseline validation).

## Council verdict (passed through 3 adversarial review rounds)
1. **No disqualifiers** against PS-26172: open-source only (TFLM+ESP-IDF), custom keyword "Pinaka", no pre-trained globals, no on-device transformers. Full mapping in [DISQUALIFIER-AUDIT.md](DISQUALIFIER-AUDIT.md).
2. **Float MFCC on Xtensa LX6 (no FPU)** is the #1 empirical risk to the <10% CPU budget — spike in P2 with documented fallbacks.
3. **MEASUREMENTS.md** added as the single source of truth for all empirical claims; no metric is an assumption.
4. **Model-readiness gate** added: `inference.h` contract exercised with dummy tensor before real `pinaka_int8.tflite` exists.
5. **DTW** is **local-only** CPU work, not "offline"; template length capped and measured.

## Wake Word & Fallback
*   **Wake word:** "Pinaka" (strong P-N-K spectrogram spikes + Indian aerospace heritage).
*   **Emergency fallback:** "SYSTEM-ABORT" via DTW (separate from neural network → no shared failure modes).
*   **No Virtual Classes:** C-style `inference.h` contracts, compile-time backend selection (avoids vtable overhead).
*   **Float MFCC:** Standard float on host for bit-parity with PC `.wav` testing; quantization at the TFLM boundary only.

## Pre-Roll Buffer Pivot
500 ms continuous circular buffer placed *before* the VAD gate so the first 50-80 ms of the wake word are never clipped while the VAD triggers. (8,000 samples × 2 B = 16 KB, not 16,000 bytes — verified.)

## Presentation Strategy
EdgeWake as COTS technology demonstrator validating aerospace principles (algorithmic diversity, strict resource budgeting), not claiming flight certification.

## Team Roles
*   **Harshit:** live telemetry dashboard.
*   **Ravi:** hardware demonstration + camera for SIH submission.
