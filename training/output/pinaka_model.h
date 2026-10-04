#ifndef PINAKA_MODEL_H
#define PINAKA_MODEL_H

extern const unsigned char pinaka_model_data[];
extern const unsigned int pinaka_model_data_len;

#define PINAKA_MODEL_SIZE  36896
#define PINAKA_SR          16000
#define PINAKA_N_MELS      40
#define PINAKA_N_FFT       512
#define PINAKA_HOP         320
#define PINAKA_FRAMES      49
#define PINAKA_AUDIO_LEN   16000

#endif
