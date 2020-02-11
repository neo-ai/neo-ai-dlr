#include "counter/model_exec_counter.h"
std::mutex g_dict;

void ModelExecCounter::update_model_run_count(std::string model) {
  ModelExecCounter::handle_dict(APPEND, model);
}

void ModelExecCounter::handle_dict(int operation, std::string model) {
  g_dict.lock();
  if (operation == APPEND) {
    std::map<std::string, int>::iterator res =  model_dict.find(model);
    if (res != model_dict.end()) {
      int count = res->second + 1;
      model_dict[model] = count;
    } else {
      model_dict[model] = 1;
    }
  } else if (operation == GETINTERIMDIC) {
    inter_dict = model_dict;
    model_dict.clear();
  }
  g_dict.unlock();
}

std::map<std::string, int > ModelExecCounter::model_dict = {};
std::map<std::string, int > ModelExecCounter::inter_dict = {};
