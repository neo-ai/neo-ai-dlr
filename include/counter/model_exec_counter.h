#ifndef MY_APPLICATION_MODEL_EXEC_COUNTER_H
#define MY_APPLICATION_MODEL_EXEC_COUNTER_H

#include<iostream>
#include<map>

using namespace std;

/*! \brief class ModelExecCounter
 */
class ModelExecCounter
{
 public:
  static void add_model_run_count(std::size_t model_hash);
  static std::map<std::size_t, int> get_model_run_count() { return model_dict; }
  static void clear_model_counts() { model_dict.clear(); }
  static std::map<std::size_t, int > model_dict;
};

#endif //MY_APPLICATION_MODEL_EXEC_COUNTER_H
