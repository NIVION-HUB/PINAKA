import os
import sys
import queue
import numpy as np
import sounddevice as sd
import librosa

# Suppress TensorFlow logging to keep the console clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
MODEL_PATH   = "training/output/pinaka_full.keras"
SAMPLE_RATE  = 16000
N_MELS       = 40
N_FFT        = 512
HOP_LENGTH   = 320

# High-sensitivity scanning: 100ms step = 10 inferences per second
WINDOW_STEP  = 0.10                          # seconds between each inference
BLOCKSIZE    = int(SAMPLE_RATE * WINDOW_STEP) # 1600 samples per block

# Multi-scale windows: catches the word whether spoken fast, normal, or slow
WINDOW_SIZES = [0.75, 1.0, 1.25]            # seconds
MAX_BUFFER   = int(max(WINDOW_SIZES) * SAMPLE_RATE)

# Confidence threshold per window
CONFIDENCE_THRESHOLD = 0.95

# Consecutive confirmations required before firing
# Set to 1 for maximum instant responsiveness
CONSECUTIVE_REQUIRED = 1

# Device index (2 = realme Buds Wireless 3 headset)
MIC_DEVICE = 2

print("\n=======================================================")
print("  PROJECT PINAKA - VIRTUAL ESP32 SIMULATOR")
print("=======================================================")
print("[*] Loading Neural Network...")

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Could not find {MODEL_PATH}. Make sure training finished!")
    sys.exit(1)

model = load_model(MODEL_PATH)
print("[*] Model loaded successfully!")

# --- FEATURE EXTRACTION (mirrors master_train.py _extract_logmel exactly) ---
def extract_features(audio_1d):
    """
    Takes a 1D float32 audio array of any length,
    pads/trims to 1 second, and returns (49, 40) Log-Mel features.
    The model always expects a 1-second clip regardless of source length.
    """
    # Pad or trim to exactly 1 second
    target = SAMPLE_RATE
    if len(audio_1d) >= target:
        audio_1d = audio_1d[:target]
    else:
        audio_1d = np.pad(audio_1d, (0, target - len(audio_1d)))

    mel = librosa.feature.melspectrogram(
        y=audio_1d, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=8000
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Min-Max normalize to [0, 1] — must match training exactly
    log_mel -= log_mel.min()
    d = log_mel.max()
    if d > 0:
        log_mel /= d

    # Transpose: (mels, frames) -> (frames, mels) = (49, 40)
    log_mel = log_mel.T
    if log_mel.shape[0] > 49:
        log_mel = log_mel[:49]
    elif log_mel.shape[0] < 49:
        log_mel = np.pad(log_mel, ((0, 49 - log_mel.shape[0]), (0, 0)))

    return log_mel


def run_multiscale_inference(raw_buffer_1d):
    """
    Runs the model on multiple window sizes simultaneously.
    Returns the highest confidence score found across all windows.
    This makes the system sensitive to fast AND slow pronunciations.
    """
    best = 0.0
    buf_len = len(raw_buffer_1d)

    for win_sec in WINDOW_SIZES:
        win_samples = int(win_sec * SAMPLE_RATE)
        if buf_len < win_samples:
            # Not enough audio yet for this window size — use what we have
            segment = raw_buffer_1d
        else:
            # Take the most recent N samples
            segment = raw_buffer_1d[-win_samples:]

        features = extract_features(segment)
        features = np.expand_dims(features, axis=(0, -1))   # (1, 49, 40, 1)
        score = float(model.predict(features, verbose=0)[0][0])
        if score > best:
            best = score

    return best


# --- AUDIO QUEUE ---
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())


# --- ADAPTIVE NOISE CALIBRATION ---
print("[*] Connecting to Microphone (Device 2: realme Buds Wireless 3)...")
print("Calibrating background noise — stay quiet for 2 seconds...")

calibration_volumes = []
try:
    with sd.InputStream(device=MIC_DEVICE, samplerate=SAMPLE_RATE,
                        channels=1, blocksize=BLOCKSIZE) as stream:
        for _ in range(int(2.0 / WINDOW_STEP)):
            chunk, _ = stream.read(BLOCKSIZE)
            calibration_volumes.append(np.sqrt(np.mean(chunk**2)))
except Exception as e:
    print(f"[WARN] Calibration failed: {e}. Using default threshold.")

baseline  = np.mean(calibration_volumes) if calibration_volumes else 0.002
SILENCE_THRESHOLD = max(baseline * 2.0, 0.001)   # 2x baseline, minimum 0.001
print(f"[OK] Baseline noise: {baseline:.5f}  |  Silence gate: {SILENCE_THRESHOLD:.5f}")

# --- MAIN LOOP ---
# Rolling raw audio buffer (holds up to max_window + 0.5s extra)
raw_samples = np.zeros(MAX_BUFFER, dtype=np.float32)

# Cooldown: after a detection, ignore audio for 2 seconds to prevent spam
COOLDOWN_BLOCKS = int(2.0 / WINDOW_STEP)
cooldown_counter    = 0
detection_count     = 0
consecutive_hits    = 0   # must hit CONSECUTIVE_REQUIRED times in a row

print("\n=======================================================")
print("[*] HIGH-SENSITIVITY EDGE AI ACTIVE")
print("[*] Scanning every 100ms across 3 window sizes (0.75s / 1.0s / 1.25s)")
print("[*] Waiting silently for Wake Word...")
print("=======================================================\n")

try:
    with sd.InputStream(device=MIC_DEVICE, samplerate=SAMPLE_RATE,
                        channels=1, callback=audio_callback, blocksize=BLOCKSIZE):
        while True:
            chunk = audio_queue.get()
            new_audio = chunk[:, 0].astype(np.float32)

            # Shift the rolling buffer and append new chunk
            raw_samples = np.roll(raw_samples, -len(new_audio))
            raw_samples[-len(new_audio):] = new_audio

            # --- Cooldown period after a detection ---
            if cooldown_counter > 0:
                cooldown_counter -= 1
                continue

            # --- Silence gate (battery saver) ---
            volume = np.sqrt(np.mean(new_audio**2))
            if volume < SILENCE_THRESHOLD:
                continue

            # --- UNIQUE FEATURE 1: HUMAN VOICE BAND FILTER ---
            # Do a lightning-fast FFT to check if the sound is in the human vocal range (300Hz - 3400Hz).
            # If it's a dog bark, AC hum, or car horn, we drop it BEFORE running the AI!
            fft_data = np.abs(np.fft.rfft(new_audio))
            freqs = np.fft.rfftfreq(len(new_audio), 1.0 / SAMPLE_RATE)
            human_band_energy = np.sum(fft_data[(freqs >= 300) & (freqs <= 3400)])
            total_energy = np.sum(fft_data)
            
            # If less than 40% of the acoustic energy is in the human voice band, it's not speech.
            if total_energy > 0 and (human_band_energy / total_energy) < 0.40:
                continue # Ignore non-human sounds, saving massive AI battery!

            # --- UNIQUE FEATURE 2: MICROSECOND LATENCY MEASUREMENT ---
            import time
            start_time = time.perf_counter()

            # --- Multi-scale inference ---
            confidence = run_multiscale_inference(raw_samples)

            # --- TEMPORAL CONSISTENCY CHECK ---
            if confidence >= CONFIDENCE_THRESHOLD:
                consecutive_hits += 1
            else:
                consecutive_hits = 0   # Reset — a single dip disqualifies it

            # --- TRIGGER (only after N consecutive high-confidence scans) ---
            if consecutive_hits >= CONSECUTIVE_REQUIRED:
                detection_count  += 1
                consecutive_hits  = 0
                
                # Stop the latency timer!
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                print(f"\n=======================================================")
                print(f"  [!!!] WAKE WORD DETECTED #{detection_count}")
                print(f"        Confidence: {confidence * 100:.1f}%  |  Confirmed x{CONSECUTIVE_REQUIRED}")
                print(f"  [SYS] ACTIVATING SPEECH AI...")
                print(f"  [NET] Opening connection to Cloud ASR...")
                print(f"  [⏱️]  Wake-to-Stream Latency: {latency_ms:.2f} ms")
                print(f"=======================================================\n")
                cooldown_counter = COOLDOWN_BLOCKS   # 2-second cooldown

except KeyboardInterrupt:
    print("\n[*] Virtual Simulator shutting down. Goodbye!")
    sys.exit(0)
except Exception as e:
    print(f"\n[ERROR] {e}")
    print("Check that your realme Buds are connected and set as input device.")
