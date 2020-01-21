#ifndef MODEL_EXEC_COUNTER_H
#define MODEL_EXEC_COUNTER_H

#include<iostream>
#include<map>

using namespace std;

/*! \brief class ModelExecCounter
 */
class ModelExecCounter
{
 public:
  static void add_model_run_count(std::string model_hash);
  static std::map<std::string, int> get_model_run_count() { return model_dict; }
  static void clear_model_counts() { model_dict.clear(); }
  static std::map<std::string, int > model_dict;
};

#endif //MODEL_EXEC_COUNTER_H
