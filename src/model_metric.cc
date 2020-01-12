#include <iostream>

#include "counter/model_metric.h"
using namespace std;

ModelMetric* ModelMetric::get_instance()
{
  if (!modelmetric) {
    modelmetric = new ModelMetric();
    if (modelmetric) {
      modelmetric->thrd = new std::thread(&ModelMetric::process_queue, modelmetric);
    }
  }
  return modelmetric;
};

ModelMetric::~ModelMetric()
{
  delete thrd;
  delete restcon;
};

void ModelMetric::process_queue()
{
  while (!stop_process) {
    std::this_thread::sleep_for(std::chrono::seconds(CALL_HOME_MODEL_RUN_COUNT_TIME_SECS));
    std::map<std::size_t, int> dict = ModelExecCounter::get_model_run_count();
    for(auto pair_dict :  dict) {
      std::string pub_data ("{");
      pub_data += "\"record_type\":";
      pub_data += std::to_string(MODEL_RUN) + ", ";
      pub_data += "\"model\":\"" + std::to_string(pair_dict.first) + "\", ";
      pub_data += "\"uuid\": \"" + device_id + "\",";
      pub_data += "\"run_count\":" + std::to_string(pair_dict.second) + "}";
      restcon->send(pub_data);
    }
    ModelExecCounter::clear_model_counts();
  }
}

ModelMetric* ModelMetric::modelmetric = nullptr;
