# TASKS.md — Execution Roadmap (36-Hour Hackathon)

## Phase P0+: Gate checks before hardware (Hours 0–3)
- [ ] **Council lock** of baseline docs (`PRD.md`, `ARCHITECTURE.md`, `RULES.md`, `DESIGN.md`, `TASKS.md`, `MEMORY.md`, `MEASUREMENTS.md`, `DISQUALIFIER-AUDIT.md`). *(This baseline is council-approved; see DISQUALIFIER-AUDIT.md.)*
- [ ] **Host parity**: write `core/` on Linux first so MFCC/VAD/kws_logic compile and run on PC `.wav` files before touching hardware.
- [ ] **Model contract gate**: stub `inference.h` + dummy int8 tensor so P3 can build `inference_tflm.cpp` *without* awaiting `pinaka_int8.tflite`. (Prevents P3 deadlock.)

## Priority 0 (P0): Audio Bring-up & Baseline (Hours 3–14)
- [ ] Wire INMP441 to ESP-WROOM-32 (SCK: 26, WS: 25, SD: 33).
- [ ] Implement `audio_i2s.cpp` with DMA buffers.
- [ ] Implement bit-shifting (24-bit to 16-bit) and DC offset removal (first-order high-pass).
- [ ] Output baseline RAM and RMS/Peak telemetry to Serial Monitor.
- [ ] Verify `fs ≈ 16000` via samples_received / elapsed sanity check.

## Priority 1 (P1): Acoustic Gate (Hours 14–18)
- [ ] Statically allocate the 16 KB `int16_t` circular buffer.
- [ ] Implement `micro_vad.c` using Exponential Moving Average (EMA).
- [ ] Verify CPU usage remains <10% while VAD is running and rejecting silence.
- [ ] Measure EMA onset latency against a clap; tune `alpha` if > 100 ms.

## Priority 2 (P2): Feature Pipeline (Hours 18–22)
- [ ] Integrate floating-point Log-Mel / MFCC extraction in `audio_features.c` (host-first).
- [ ] Validate float-MFCC frame time on ESP32 (Goal: < 15ms; measured in MEASUREMENTS.md).
- [ ] Implement INT8 quantization step (read `scale`/`zero_point` from model input tensor, not hardcoded).
- [ ] [If > 15 ms] spike: fixed-point MFCC, mel-count reduction, or ESP-DSP FFT.

## Priority 3 (P3): KWS Inference (Hours 22–26)
- [ ] Convert `pinaka_int8.tflite` to a C-array (`pinaka_int8.cc`). *(Or run via dummy-tensor gate from P0+.)*
- [ ] Implement `inference_tflm.cpp` with statically allocated Tensor Arena.
- [ ] Implement temporal debouncing and confidence gating in `kws_logic.c`.
- [ ] Measure end-to-end KWS latency.

## Priority 4 (P4): Network Handoff (Hours 26–32)
- [ ] Establish persistent Wi-Fi and WebSocket connection (boot retry loop, non-blocking).
- [ ] On wake trigger, transmit 500ms pre-roll + live PCM window.
- [ ] Implement configurable End-Of-Speech (EOS) timeout (Default 800ms).
- [ ] Measure complete Wake-to-ASR latency (`T3 − T0`).

## Priority 5 (P5): Fallback & Polish (Hours 32–36)
- [ ] Integrate constrained DTW matcher for "SYSTEM-ABORT" local-only fallback (cap template length; measure frame cost).
- [ ] Wire GPIO to trigger local LED on DTW success.
- [ ] Finalize telemetry output for Harshit to pipe into the live Node.js dashboard.
- [ ] Rehearse presentation with Ravi capturing the live hardware timeline.
