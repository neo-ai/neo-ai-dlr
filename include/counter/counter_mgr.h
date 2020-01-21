#ifndef COUNTERMGR_H
#define COUNTERMGR_H

#include <fstream>
#include <ostream>
#include <iostream>
#include <dmlc/logging.h>

#include "device_info.h"
#include "system.h"
#include "publisher.h"
#include "config.h"
#include "model_exec_counter.h"
#include "model_metric.h"
#include "helper.h"

using namespace std;

extern std::string ext_path;

/*! \brief class CounterMgr
 */
class CounterMgr {
 public:
  static CounterMgr* get_instance();
  static void release_instance() { 
    instance->msg_publisher->release_instance();
    instance->msg_publisher = nullptr;
    instance->model_metric->release_instance();
    instance->model_metric = nullptr;
    delete instance;
    instance = nullptr;
  }
  static void set_data_consent(int);
  static bool is_feature_enabled();
  bool is_device_info_published();
  void runtime_loaded();
  void model_loaded(std::string model);
  void model_run(std::string model);
 protected:
  void model_load_publish(int msg_type, string model, int count =0);
  void push(string& data) { 
    if (msg_publisher) {
      msg_publisher->send(data);
    }
  };
 private:
  CounterMgr();
  ~CounterMgr() {
    delete system;
    system = nullptr;
  }
  // fields for matric data type
  const int RUNTIME_LOAD = 1;
  const int MODEL_LOAD = 2;
  const int MODEL_RUN = 3;
  static int feature_enable;
  static int DEFAULT_ENABLE_FEATURE;
  System *system;
  static CounterMgr* instance;
  MsgPublisher* msg_publisher;
  ModelMetric* model_metric;
};

/*! \brief Hook for Call Home Feature.
 */
extern CounterMgr *instance;
inline void CallHome(int type, std::string model= std::string())
{
  CounterMgr* instance;
  if (!instance) {
    #if defined(__ANDROID__)
    instance = CounterMgr::get_instance();
    if (!instance)  {
      LOG(FATAL) << "Call Home Feature not initialize!";
      return;
    }
    #else
    return;
    #endif
  }
  switch (type) {
    case 1:
      instance->runtime_loaded();
      break;
    case 2:
      instance->model_loaded(model);
      break;
    case 3:
      instance->model_run(model);
      break;
    case 0:
      instance->release_instance();
      break;
  }
}
 

#endif //COUNTERMGR_H
