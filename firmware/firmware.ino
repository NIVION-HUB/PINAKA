/*
 ==============================================================================
  PROJECT PINAKA - ESP32 FIRMWARE
  ISRO SIH-26172
 ==============================================================================
*/

#include <Arduino.h>
#include <driver/i2s.h>

// Include the compiled AI Model
#include "pinaka_model.h"

// TensorFlow Lite Micro Headers
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ==========================================
// PIN DEFINITIONS & CONSTANTS
// ==========================================
// INMP441 I2S Pins
#define I2S_WS 15  // L/R Clock
#define I2S_SD 32  // Data Out
#define I2S_SCK 14 // Serial Clock

#define SPEECH_AI_LED 2 // Built-in LED on ESP32

// Audio Configuration (Matched to Python pipeline)
#define SAMPLE_RATE 16000
#define BUFFER_SAMPLES 512

// ==========================================
// TENSORFLOW LITE VARIABLES
// ==========================================
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor Arena (Memory for the model to run)
constexpr int kTensorArenaSize = 60 * 1024; // 60 KB SRAM for inference
uint8_t tensor_arena[kTensorArenaSize];

// ==========================================
// I2S MICROPHONE SETUP
// ==========================================
void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // INMP441 outputs 32-bit (we use 16-bit shift)
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = BUFFER_SAMPLES,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

// ==========================================
// INITIALIZATION
// ==========================================
void setup() {
  Serial.begin(115200);
  pinMode(SPEECH_AI_LED, OUTPUT);
  digitalWrite(SPEECH_AI_LED, LOW);

  Serial.println("=========================================");
  Serial.println("  PROJECT PINAKA - INITIALIZING...       ");
  Serial.println("=========================================");

  // 1. Initialize Microphone (DMA)
  setupI2S();
  Serial.println("[OK] I2S Microphone DMA started.");

  // 2. Load TFLite Model
  model = tflite::GetModel(pinaka_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Model schema version mismatch!");
    while(1);
  }

  // 3. Setup Ops Resolver & Interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, nullptr);
  interpreter = &static_interpreter;

  // 4. Allocate Memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() failed");
    while(1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("[OK] Neural Network loaded into IRAM.");
  Serial.println("[OK] System Armed. Listening for 'PINAKA'...");
}

// ==========================================
// MAIN INFERENCE LOOP
// ==========================================
void loop() {
  int32_t raw_samples[BUFFER_SAMPLES];
  size_t bytes_read = 0;

  // Read audio from I2S DMA
  i2s_read(I2S_NUM_0, &raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);

  // TODO: Extract Log-Mel Spectrogram here (DSP logic)
  // For now, we simulate feeding data to the input tensor:
  /*
  for (int i = 0; i < PINAKA_N_MELS * PINAKA_FRAMES; i++) {
    input->data.int8[i] = generated_spectrogram_val;
  }
  */

  // Run Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[ERROR] Inference failed!");
    return;
  }

  // Check Output Score
  int8_t prediction = output->data.int8[0]; 
  
  // Convert INT8 output (-128 to 127) to percentage (0 to 100)
  float confidence = ((prediction + 128) / 255.0) * 100.0;

  if (confidence > 90.0) {
    Serial.println("\n=========================================");
    Serial.println("🎙️ WAKE WORD DETECTED! ACTIVATING SPEECH AI...");
    Serial.printf("🎙️ Confidence: %.2f%%\n", confidence);
    Serial.println("=========================================\n");
    
    // Trigger LED/Relay
    digitalWrite(SPEECH_AI_LED, HIGH);
    delay(2000); 
    digitalWrite(SPEECH_AI_LED, LOW);
  }
}
