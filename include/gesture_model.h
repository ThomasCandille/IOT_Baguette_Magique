#pragma once

// TensorFlow Lite Micro : entrée INT8 [1, 75, 4], ordre X, Y, GX, GY.
// Appliquer la même normalisation et la même cadence que le dataset.
// Quantification : clamp(round(valeur_normalisee / scale) + zero_point, -128, 127).
extern const unsigned char g_gesture_model[];
extern const unsigned int g_gesture_model_len;
constexpr int kGestureSamples = 75;
constexpr int kGestureChannels = 4;
constexpr int kGestureClasses = 4;
constexpr float kGestureInputScale = 3.9215688594e-03f;
constexpr int kGestureInputZeroPoint = -128;
constexpr float kGestureOutputScale = 8.7279252708e-02f;
constexpr int kGestureOutputZeroPoint = 22;
constexpr const char* kGestureLabels[] = {"circle", "hor", "nothing", "ver"};
