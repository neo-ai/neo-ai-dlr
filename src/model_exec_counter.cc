#include "counter/model_exec_counter.h"

void ModelExecCounter::add_model_run_count(std::size_t model_hash)
{
  std::map<std::size_t, int>::iterator res =  model_dict.find(model_hash);
  if (res != model_dict.end()) {
    int count = res->second + 1;
    model_dict[model_hash] = count;
  } else {
    model_dict[model_hash] = 1;
  }
}

std::map<std::size_t, int > ModelExecCounter::model_dict = {};
