# RULES.md

## 1. Architectural Boundaries
*   **No Hardware Leaks in Core:** Files inside `/core` must never `#include` ESP-IDF headers, FreeRTOS libraries, or Arduino functions. They must compile on a standard Linux GCC environment.
*   **Inference Abstraction:** `core/kws_logic.c` must never directly invoke a TFLite MicroInterpreter. It must call the `kws_infer()` contract defined in `core/inference.h`.
*   **Boundary Contract:** `core/inference.h` is a C-compatible header (`extern "C"` when compiled as C++). The platform `.cpp` files implement it; the core `.c` files consume it.

## 2. Memory Management
*   **No Dynamic Allocation in Hot Paths:** The audio ingestion pipeline must never use `malloc()` or `new`.
*   **Static Buffers Only:** The 16 KB pre-roll buffer and the TFLM Tensor Arena must be statically allocated in `.bss` at boot.
*   **Metrics Over Guesses:** Do not guess RAM usage. Use `esp_get_free_heap_size()` and `esp_get_minimum_free_heap_size()` strictly after each new subsystem is initialized. See [MEASUREMENTS.md](MEASUREMENTS.md).

## 3. Mathematical & Acoustic Standards
*   **Audio Format:** Strictly 16,000 Hz, 16-bit Mono PCM.
*   **Float is Permitted:** Use standard `float` for MFCC feature extraction to ensure bit-parity with PC-based `.wav` testing. Quantize to `INT8` strictly at the boundary of the neural network input tensor. (See risk note in [DISQUALIFIER-AUDIT.md](DISQUALIFIER-AUDIT.md) re: LX6 without FPU.)
*   **DC Offset Correction:** All raw INMP441 I2S reads must pass through a first-order high-pass filter before reaching the VAD or Circular Buffer.

## 4. Networking
*   **Persistent Connections:** Sockets must be opened at boot with a non-blocking retry loop (boot must not stall if Wi-Fi/AP is unavailable at t0).
*   **Core Pinning:** WebSocket transmissions must execute on Core 0 (Network) to prevent blocking the I2S DMA reads on Core 1 (Audio/ML).

## 5. Algorithmic Diversity (Terminology)
*   The "local fallback" DTW matcher is **local-only** (does not require cloud), not "offline" in the sense of precomputed. It is **CPU work**, so its frame-slice cost is measured in [MEASUREMENTS.md](MEASUREMENTS.md) and its template length is capped to protect the <10% CPU budget.

## 6. Submission Discipline
*   Never claim efficiency, accuracy, or latency. Only record measured values in [MEASUREMENTS.md](MEASUREMENTS.md).
