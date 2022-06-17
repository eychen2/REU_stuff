#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "model-final.h"
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  constexpr int kTensorArenaSize=1024*20;
  uint8_t tensor_arena[kTensorArenaSize];
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}
const float temp[784] = {-95,-78,-61,-44,0,2,0,4,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,1,0,2,-51,105,0,3,-1,30,0,0,0,70,0,0,0,70,2,26,-59,1,0,0,2,26,-59,2,0,0,8,0,69,0,0,52,-48,-37,64,0,32,6,-57,13,1,2,-88,-106,1,1,24,66,1,-69,33,97,-95,-73,-114,80,36,-51,37,-9,-128,17,90,-128,-40,-85,0,0,1,1,8,10,22,-101,90,-9,5,53,108,5,99,-20,22,-115,0,2,-51,105,0,3,-1,-29,0,0,0,70,0,0,0,70,2,26,-59,2,0,0,2,26,-59,1,0,0,8,0,69,0,0,52,-96,-124,64,0,32,6,-9,100,1,1,24,66,1,2,-88,-106,33,97,1,-69,36,-51,37,-9,-95,-73,-114,81,-128,17,96,40,-110,27,0,0,1,1,8,10,5,73,-84,-40,22,-101,90,-9,-10,108,34,-40,0,2,-51,105,0,4,0,22,0,0,0,70,0,0,0,70,2,26,-59,1,0,0,2,26,-59,2,0,0,8,0,69,0,0,52,-48,36,64,0,32,6,-57,-60,1,2,-88,-106,1,1,24,66,1,-69,33,97,-95,-73,-114,81,36,-51,37,-8,-128,16,90,-128,-112,7,0,0,1,1,8,10,22,-101,98,-77,5,73,-84,-40,-24,-66,-55,-124,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
void setup() {
  Serial.begin(4800);
  while (!Serial);
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_final_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddQuantize();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddRelu();
  resolver.AddLogistic();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
   // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
  for(unsigned int i=0; i<784;++i)
    input->data.int8[i]=temp[i]/ input->params.scale + input->params.zero_point;
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
  // Obtain the quantized output from model's output tensor
  int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  float y = (y_quantized - output->params.zero_point) * output->params.scale;
  Serial.println(output->data.int8[0]);
  Serial.println(y, 4);
  
}


void loop() {
      
}
