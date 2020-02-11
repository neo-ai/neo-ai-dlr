#include <iostream>

#include "counter/model_metric.h"
using namespace std;

ModelMetric* ModelMetric::get_instance() {
  if (!modelmetric) {
    modelmetric = new ModelMetric();
    if (modelmetric) {
      modelmetric->thrd = new std::thread(&ModelMetric::process_queue, modelmetric);
    }
  }
  return modelmetric;
};

ModelMetric::~ModelMetric() {
  delete thrd;
  thrd = nullptr;
};

void ModelMetric::process_queue() {
  while (!stop_process) {
    std::this_thread::sleep_for(std::chrono::seconds(CALL_HOME_MODEL_RUN_COUNT_TIME_SECS));
    std::map<std::string, int> dict = ModelExecCounter::get_dict();
    for(auto pair_dict :  dict) {
      char buff[128];
      snprintf(buff, sizeof(buff), "{ \"record_type\": %s, \"model\": \"%s\", \"uuid\": \"%s\", \"run_count\": \"%s\" }", std::to_string(MODEL_RUN).c_str(), pair_dict.first.c_str(), device_id.c_str(), std::to_string(pair_dict.second).c_str());
      std::string pub_data = buff;
      publisher->send(pub_data);
    }
  }
}

ModelMetric* ModelMetric::modelmetric = nullptr;
