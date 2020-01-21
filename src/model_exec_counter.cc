#include "counter/model_exec_counter.h"

void ModelExecCounter::add_model_run_count(std::string model_hash)
{
  std::map<std::string, int>::iterator res =  model_dict.find(model_hash);
  if (res != model_dict.end()) {
    int count = res->second + 1;
    model_dict[model_hash] = count;
  } else {
    model_dict[model_hash] = 1;
  }
}

std::map<std::string, int > ModelExecCounter::model_dict = {};
