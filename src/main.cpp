#include <Arduino.h>
#include <LiquidCrystal.h>
#include <Wire.h>
#include <cmath>
#include "gesture_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// LCD : RS, E, D4, D5, D6, D7. RW doit être relié à GND.
LiquidCrystal lcd(1, 2, 3, 4, 5, 8);
constexpr int kSda = 7, kScl = 6;
constexpr uint32_t kSamplePeriodMs = 20;  // 50 Hz, 75 mesures = 1,5 s.

// Bornes de dataset_thomas_normalised/normalisation.json : X, Y, GX, GY.
constexpr float kMinimum[] = {-2.0f, -2.0f, -250.13741f, -250.13741f};
constexpr float kMaximum[] = {1.99994f, 1.99994f, 250.12978f, 250.12978f};
uint8_t mpuAddress = 0x68;
alignas(16) uint8_t tensorArena[64 * 1024];
tflite::MicroMutableOpResolver<8> resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void show(const char* first, const char* second = "") {
  lcd.clear();
  lcd.print(first);
  lcd.setCursor(0, 1);
  lcd.print(second);
}

void stop(const char* message) {
  show("Erreur", message);
  Serial.println(message);
  while (true) delay(1000);
}

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(mpuAddress);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readRegisters(uint8_t reg, uint8_t* data, uint8_t length) {
  Wire.beginTransmission(mpuAddress);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(mpuAddress, length) != length) return false;
  for (int i = 0; i < length; ++i) data[i] = Wire.read();
  return true;
}

bool readMotion(float values[4]) {
  uint8_t data[14];
  if (!readRegisters(0x3B, data, sizeof(data))) return false;
  // Permutation gyro : nouveau X = ancien Z, Y = ancien X, Z = ancien Y.
  // Le modèle utilise X, Y, GX, GY : lire les gyros Z et X du capteur.
  const int offsets[] = {0, 2, 12, 8};
  for (int axis = 0; axis < 4; ++axis) {
    int i = offsets[axis];
    int16_t raw = static_cast<int16_t>((data[i] << 8) | data[i + 1]);
    values[axis] = raw / (axis < 2 ? 16384.0f : 131.0f);  // g et degrés/s.
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  lcd.begin(16, 2);
  show("Initialisation");
  Wire.begin(kSda, kScl);
  Wire.setClock(400000);
  Wire.setTimeOut(100);

  // MPU6050 : essayer les deux adresses possibles.
  uint8_t identity = 0;
  if (!readRegisters(0x75, &identity, 1) || identity != 0x68) {
    mpuAddress = 0x69;
    if (!readRegisters(0x75, &identity, 1) || identity != 0x68) stop("MPU6050 absent");
  }
  if (!writeRegister(0x6B, 0x01)) stop("Config MPU6050");  // Réveil.
  delay(100);
  if (!writeRegister(0x1A, 0x03) ||  // Filtre passe-bas, horloge à 1 kHz.
      !writeRegister(0x19, 19) ||    // 1000 / (19 + 1) = 50 Hz.
      !writeRegister(0x1B, 0x00) ||  // Gyroscope : +/-250 degrés/s.
      !writeRegister(0x1C, 0x00))    // Accéléromètre : +/-2 g.
    stop("Config MPU6050");

  const tflite::Model* model = tflite::GetModel(g_gesture_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) stop("Version modele");
  resolver.AddReshape();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddStridedSlice();
  resolver.AddMean();
  resolver.AddConcatenation();
  resolver.AddTranspose();
  resolver.AddFullyConnected();
  static tflite::MicroInterpreter runner(model, resolver, tensorArena, sizeof(tensorArena));
  interpreter = &runner;
  if (interpreter->AllocateTensors() != kTfLiteOk) stop("Memoire modele");
  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
      input->dims->size != 3 || input->dims->data[0] != 1 ||
      input->dims->data[1] != kGestureSamples || input->dims->data[2] != 4 ||
      output->dims->size != 2 || output->dims->data[0] != 1 ||
      output->dims->data[1] != kGestureClasses || input->params.scale <= 0)
    stop("Format modele");
  Serial.printf("Modele charge : %u octets, arena : %u octets\n",
                g_gesture_model_len, static_cast<unsigned>(runner.arena_used_bytes()));
}

void loop() {
  for (int seconds = 3; seconds > 0; --seconds) {
    show("Preparez le geste");
    lcd.print(seconds);
    delay(1000);
  }
  show("Faites le geste", "Capture 1.5 s");
  uint32_t nextSample = millis();
  for (int sample = 0; sample < kGestureSamples; ++sample) {
    float values[4];
    if (!readMotion(values)) stop("Lecture MPU6050");
    for (int axis = 0; axis < 4; ++axis) {
      float normalized = (values[axis] - kMinimum[axis]) / (kMaximum[axis] - kMinimum[axis]);
      normalized = constrain(normalized, 0.0f, 1.0f);
      int quantized = lroundf(normalized / input->params.scale) + input->params.zero_point;
      input->data.int8[sample * 4 + axis] = constrain(quantized, -128, 127);
    }
    nextSample += kSamplePeriodMs;
    while (static_cast<int32_t>(millis() - nextSample) < 0) delay(1);
  }
  if (interpreter->Invoke() != kTfLiteOk) stop("Reconnaissance");
  int predicted = 0;
  for (int i = 1; i < kGestureClasses; ++i)
    if (output->data.int8[i] > output->data.int8[predicted]) predicted = i;
  show("Signe detecte :", kGestureLabels[predicted]);
  Serial.printf("Signe : %s\n", kGestureLabels[predicted]);
  delay(3000);
}
