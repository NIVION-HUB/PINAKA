# MEASUREMENTS.md — Empirical Validation Log

**Principle:** every number below starts as a **hypothesis**. It is not architecture until the ESP32 proves it. If a hypothesis fails, the *design* changes — never the measurement.

## 1. P0 Baseline (audio bring-up, diagnostics)

| Hypothesis | How to measure | Tool | Target (pass) | Result |
|---|---|---|---|---|
| `fs ≈ 16000 Hz` | samples_received / elapsed_time over a known wall-clock interval | serial, 10 s gate | effective_fs in [15840, 16160] Hz | [Pending] |
| `dma_overruns == 0` | counter in the I2S DMA callback | serial | 0 | [Pending] |
| `rms reacts to sound` | moving RMS of buffered PCM, in dBFS | serial / oscilloscope-style log | rises > -20 dBFS when speaking | [Pending] |
| `dc_offset ≈ 0` | mean of buffered PCM after high-pass filter | serial | | 0.000 after filter |
| `clip == 0` | count of samples at ±32767 (or 0x7FFF) | serial | 0 or rare | [Pending] |
| `peak < 0x7FFF margin` | max | 16-bit | no sustained saturation | [Pending] |
| `heap_min > 40 KB` after I2S init | `esp_get_minimum_free_heap_size()` | serial | > 40 KB | [Pending] |

## 2. P1 VAD (Gaussian-EMA noise floor)

| Hypothesis | How to measure | Target | Result |
|---|---|---|---|
| EMA noise floor adapts without hunting | track `vad_noise_floor` over minutes of silence | stable, no oscillation at ±2 dB | [Pending] |
| Idle CPU with VAD running stays < 10% | xPortGetIdleHertz() or core frequency counter over 60 s | < 10% | [Pending] |
| `alpha = 0.05` onset latency | measure seconds from a sharp clap to VAD `SPEECH` flag | < 100 ms | [Pending] |

*If onset latency > 100 ms, raise `alpha` toward 0.12–0.2 and re-measure.*

## 3. P2 Feature pipeline

| Hypothesis | How to measure | Target | Result |
|---|---|---|---|
| `MFCC frame (30 ms, 40 mel, log, DCT) < 15 ms` on Xtensa LX6 | `xTaskGetTickCount()` / cycle counter around extraction | < 15 ms | [Pending] |
| float-vs-fixed-point parity | same input PCM → identical MFCC to 3 decimals | bit-identical | [Pending] |
| 20 ms stride → feature rate ≈ 50 Hz | feature_count / elapsed | ≈ 50 Hz | [Pending] |

*If MFCC frame > 15 ms, choose before P3: (a) fixed-point MFCC, (b) 26–32 mel filters, or (c) ESP-DSP-optimized FFT.*

## 4. P3 KWS / quantization

| Hypothesis | How to measure | Target | Result |
|---|---|---|---|
| `input scale` and `zero_point` read from model input tensor (not hardcoded) | print values from `interpreter->input(0)->params` | match `pinaka_int8.tflite` | [Pending] |
| quantized range stays within [-128, 127] | clip check over 60 s of audio | 0 overflow clips | [Pending] |
| KWS inference time per frame | cycle counter | < 5 ms | [Pending] |
| `pre_roll = 8000 samples = 16 KB` | sizeof(buffer) == 16384, 500 ms coverage | verified | [Verified] |

## 5. P4–P5 Network / fallback

| Hypothesis | How to measure | Target | Result |
|---|---|---|---|
| `wake-to-ASR latency (T3 − T0)` | timestamp at keyword-end → first pre-roll byte received | minimal, logged per wake | [Pending] |
| `false activations ≈ near-zero` | count spurious SPEECH flags per hour of silence | < 1 / hour | [Pending] |
| `DTW "SYSTEM-ABORT" template latency` | frame-slice cost of one template match | < 50 ms / frame | [Pending] |
| Peak total RAM (boot → idle → worst-case wake) | `esp_get_minimum_free_heap_size()` from boot | < 256 KB | [Pending] |

## Notes on measurement discipline
1. Never average away a failure. Log the *minimum* heap and the *maximum* frame time, not the mean.
2. Record which hardware revision / ESP-IDF version produced each result — the <10% CPU and <256 KB numbers are hardware-specific.
3. `MEASUREMENTS.md` is a living artifact; commit it after every milestone so regressions are visible.
