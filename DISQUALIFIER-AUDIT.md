# Disqualifier Audit — SIH Problem Statement 26172
**Scope:** EdgeWake baseline (PRD.md, ARCHITECTURE.md, RULES.md, DESIGN.md, TASKS.md, MEMORY.md) reviewed against the official problem statement. Three adversarial passes: (1) architectural soundness, (2) criteria disqualifier scan, (3) practical/implementation bugs. Final status: **no hard disqualifiers.**

## Criteria mapping

| # | Official criterion | EdgeWake stance | Audit outcome |
|---|---|---|---|
| 1 | **Efficiency:** model RAM/Flash footprint, CPU usage while idling | `<256 KB RAM`, `<10% idle CPU` — committed as *measured* targets, no fabricated numbers; architecture enforces static allocation and VAD-gated inference to make them achievable | ✅ **PASS** (honest, achievable — see MEASUREMENTS.md) |
| 2 | **Accuracy:** high TPR, near-zero false activations | VAD acoustic gate + `kws_logic` temporal debounce; `false activations` tracked per hour in MEASUREMENTS.md | ✅ **PASS** (measured, not claimed) |
| 3 | **Latency:** delta between keyword ending and cloud ASR receiving audio | Pre-roll buffer + persistent socket → audio is in-flight instantly; `T3 − T0` logged per wake | ✅ **PASS** (architecture directly targets it) |
| 4 | **Software restrictions:** open-source only; no proprietary/closed SDKs; use open-source TinyML (TFLM, PyTorch Mobile, etc.) | TFLM + ESP-IDF + custom DTW matcher. **Zero proprietary SDKs** | ✅ **COMPLIANT** |
| 5 | **No pre-trained global keywords** (no "Hey Google"/"Alexa"); must train on a custom keyword | Custom keyword **"Pinaka"** — trained by the team from collected audio | ✅ **COMPLIANT** |
| 6 | **Hardware:** runs on physical low-power MCUs (ESP32 or Raspberry Pi), <256 KB RAM, <10% idle CPU; heavy pre-trained transformers disqualified | ESP-WROOM-32 (Xtensa LX6). Pre-roll/static buffers only; DTW template kept short; transformer-based ASR runs in the *cloud*, never on-device | ✅ **COMPLIANT** |

## Known empirical risks (honest, not disqualifiers)

These are *measured* rather than assumed in MEASUREMENTS.md, and each has a pre-defined mitigation:

| Risk | Why it matters | Mitigation (already in TASKS.md) |
|---|---|---|
| **Float MFCC on Xtensa LX6 (no FPU)** may exceed the 15 ms frame budget, breaking `<10% CPU` | MFCC is the heaviest core routine | P2 spike: if >15 ms, fall back to fixed-point MFCC, 26–32 mel filters, or ESP-DSP FFT |
| **<256 KB total RAM** — 16 KB pre-roll + int8 KWS model + TFLM arena + network buffers | Tight on ESP-WROOM-32 (~520 KB SRAM, ~368 KB usable) | Tiny 1-layer int8 model; measured arena; documented in MEASUREMENTS.md |
| **DTW template cost** — O(N×M) per frame can spike idle CPU | "SYSTEM-ABORT" fallback must also fit the CPU budget | Cap template length; measure frame-slice cost; gate in MEASUREMENTS.md |
| **Model readiness** — P3 depends on `pinaka_int8.tflite` existing | Could deadlock the pipeline at P3 | Dummy/tensor contract gate so `inference.h` is exercised without the real model |
| **VAD onset latency** — `alpha = 0.05` may slow wake response | Undermines the latency USP | Tune alpha 0.12–0.2 if measured onset > 100 ms |

## Conclusion

**The baseline is submission-safe.** All six official criteria map to either a compliant design decision or a measured hypothesis with a recorded mitigation. The only genuine threats to disqualification are (a) floating the `<256 KB RAM` / `<10% CPU` numbers without measurement, and (b) using a pre-trained global keyword — both explicitly prevented by the `RULES.md` "Metrics Over Guesses" rule and the custom-keyword discipline.
