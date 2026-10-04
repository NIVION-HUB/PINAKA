# ARCHITECTURE.md

## 1. System Topology

```text
                               EDGEWAKE
                                   │
                          INMP441 / I2S DMA
                                   │
                        Continuous 16-kHz int16 PCM
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
    Circular Pre-Roll PCM Buffer              Adaptive Acoustic VAD
    (Retains recent 500ms context)                      │
              │                                         │
              │                              ┌──────────┴──────────┐
              │                              │                     │
              │                           INACTIVE              SPEECH
              │                        (No Feature/KWS)            │
              │                              │                     │
              └──────────────────────────────┼─────────────────────┘
                                             ▼
                                 Pre-Roll + Live PCM Window
                                             │
                                       MFCC / Log-Mel
                                          (float)
                                             │
                                       float → INT8
                                             │
                                    Tiny INT8 KWS Model
                                (Confidence + Temporal Logic)
                                             │
                                          PINAKA
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             Pre-Roll + Live Audio                          Local Fallback
                       │                                           │
             Persistent WebSocket                           DTW / GPIO Demo
                       │
                       ▼
                   Remote ASR

```

## 2. Directory Structure & Hardware Isolation

The repository enforces a strict boundary between platform-independent DSP/ML logic and hardware-specific drivers.

```text
edgewake/
├── core/                       # ZERO hardware or OS dependencies
│   ├── audio_features.c        # Float MFCC/Log-Mel logic
│   ├── micro_vad.c             # EMA noise floor tracking
│   ├── kws_logic.c             # Confidence thresholds & temporal debounce
│   ├── emergency_matcher.c     # DTW template logic
│   └── inference.h             # C-API boundary for neural inference
│
├── platform/
│   ├── esp32/                  # ESP-WROOM-32 Implementation
│   │   ├── audio_i2s.cpp       # I2S DMA configuration
│   │   ├── network_ws.cpp      # Persistent socket & pre-roll transmission
│   │   ├── inference_tflm.cpp  # TensorFlow Lite for Microcontrollers
│   │   └── system_metrics.cpp  # Heap, CPU, and latency instrumentation
│   │
│   └── linux/                  # Rapid-Testing Backend
│       ├── audio_wav.cpp       # File reader for PC testing
│       └── inference_tflite.cpp# Desktop TFLite invocation
│
└── model/
    ├── pinaka_int8.tflite
    └── pinaka_int8.cc          # Hex array for ESP32 flash

```

## 3. Boundary Contract
*   `core/` files must never `#include` ESP-IDF, FreeRTOS, or Arduino headers. They compile on plain Linux GCC.
*   `core/kws_logic.c` must never call a TFLite MicroInterpreter directly. It calls `kws_infer()` defined in `core/inference.h`.
*   `inference.h` is a C-compatible header (`extern "C"` when compiled as C++).