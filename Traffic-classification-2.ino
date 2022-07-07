#include <ArduinoJson.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "rpcWiFi.h"
#include "model-final.h"
using namespace std;
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  constexpr int kTensorArenaSize=1024*10;
  uint8_t tensor_arena[kTensorArenaSize];
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}
const char* ssid = "FIU_WiFi";
const char* password = "ddotdd13";
const char* host ="10.110.195.35";
const uint16_t port = 5000;
//const char* server = "10.110.195.35:5000";

String data;
WiFiClient client;
float s_input[784] = {};
int label =-1;
void setup() {
  Serial.begin(9600);
  while (!Serial);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
  Serial.println("Connecting to WiFi..");
  WiFi.begin(ssid);
  while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.println("Connecting to WiFi..");
        WiFi.begin(ssid);
    }
  Serial.println("Connected to the WiFi network");
  Serial.print("IP Address: ");
  Serial.println (WiFi.localIP()); // prints out the device's IP address */
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  Serial.println("reached 1");
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
  Serial.println("reached 2");
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
Serial.println("reached 3");
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
   // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("reached 4");
  /*float value=0;
  int count=0;
  string test;
  for(unsigned int i=0; i<1000;++i)
  {
    for(unsigned int j=0; j<784;++j)
    {
      getline(file1, test);
      value = stof(test);
      Serial.println(value);                             
      delay(100);
      input->data.int8[j]=value/ input->params.scale + input->params.zero_point;
      delay(100);
    }
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
    }
    int8_t y_quantized = output->data.int8[0];
    float y = (y_quantized - output->params.zero_point) * output->params.scale;
    if(y>=0.5)
     check=1;
    else
      check=0;
     getline(file2, test);
     label = stoi(test);
    if(check==label)
      ++count;
  }
  Serial.print("Number of test data predicted correctly: ");
  Serial.println(count);*/
  /*int count=0;
  int index=0;
  for(unsigned int a=0; a<5;++a)
  {
  long startMicros = millis();
  for(unsigned int i=0; i<40;++i)
  {
    for(unsigned int j=0; j<784;++j)
    {
      input->data.int8[j]=inputs[index+j]/ input->params.scale + input->params.zero_point;
    }
    index+=784;
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
    }
    int8_t y_quantized = output->data.int8[0];
    float y = (y_quantized - output->params.zero_point) * output->params.scale;
    if(y>=0.5)
     check=1;
    else
      check=0;
    if(check==labels[i])
      ++count;
  }
  long endMicros = millis();
  Serial.println("Execution time in ms: ");
  Serial.println(endMicros - startMicros);
  }
  
  Serial.print("Number of test data predicted correctly: ");
  Serial.println(count);*/
  Serial.println("End of setup");
}
void loop() {
  Serial.println("In loop");
  int check=-1;
  bool state=true;
  if(state)
  {
    Serial.println("reached");
    getData();
  }
  for(unsigned int j=0; j<784;++j)
  {
    input->data.int8[j]=s_input[j]/ input->params.scale + input->params.zero_point;
  }
   TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
  
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
    }
    int8_t y_quantized = output->data.int8[0];
    float y = (y_quantized - output->params.zero_point) * output->params.scale;
    if(y>=0.5)
     check=1;
    else
      check=0;
    if(check==label)
      Serial.println("The model predicted correctly");
    else
      Serial.println("The model predicted incorrectly");
      delay(10000);    
}
void getData(){
  Serial.println("Made it into the function");
  if (!client.connect(host, port)) {
        Serial.println("Connection failed.");
        Serial.println("Waiting 5 seconds before retrying...");
        delay(5000);
        return;
    }
    client.print("GET /data HTTP/1.1\n\n");
    int maxloops = 0;
 
    //wait for the server's reply to become available
    while (!client.available() && maxloops < 1000) {
        maxloops++;
        delay(1); //delay 1 msec
    }
    if (client.available() > 0) {
        //read back one line from the server
        data = client.readString(); // Read from the server response
        // Proceed various line-endings
        data.replace("\r\n", "\n");
        data.replace('\r', '\n');
        data.replace("\n", "\r\n");
        Serial.println(data);
    } else {
        Serial.println("client.available() timed out ");
    }
  /*if (!client.connect(server, 4000)) {
        Serial.println("Connection failed!");
    }
    else {
        Serial.println("Connected to server!"); 
        // Make a HTTP request:
        Serial.println("Reached checkpoint 1");
        client.println("GET http://10.110.195.35:5000/data HTTP/1.1");
        client.println("Host: http://10.110.195.35:5000");
        client.println("Connection: close");
        client.println();
        Serial.println("Reached checkpoint 2");
 
        while (client.connected()) {
            String line = client.readStringUntil('\n');
            if (line == "\r") {
                Serial.println("headers received");
                break;
            }
        }
        Serial.println("Reached checkpoint 3");
        while(client.available())
        {
          String line = client.readStringUntil('\r');
          data = line;
        }
        Serial.println(data);
        client.stop();
        Serial.println("closing connection");
    }*/
 
    DynamicJsonDocument doc(16384);
    deserializeJson(doc, data);
    for(unsigned int i=0; i<784;++i)
      s_input[i] = doc["data"][i];
    label = doc["label"];
}
