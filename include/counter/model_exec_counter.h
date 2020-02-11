#ifndef MODEL_EXEC_COUNTER_H
#define MODEL_EXEC_COUNTER_H

#include<iostream>
#include<map>
#include <mutex>

using namespace std;
enum record { RUNTIME_LOAD=1, MODEL_LOAD=2, MODEL_RUN=3}; 

/*! \brief class ModelExecCounter
 */
class ModelExecCounter
{
 public:
  static void update_model_run_count(std::string model);
  static void handle_dict(int operation, std::string model = std::string()); 
  static std::map<std::string, int> get_dict() { 
    handle_dict(GETINTDIC); 
    return inter_dict; 
  }
  static std::map<std::string, int > model_dict;
  static std::map<std::string, int > inter_dict;
  static const int APPEND = 1;
  static const int GETINTDIC = 2;
};

#endif //MODEL_EXEC_COUNTER_H
