#ifndef COUNTERMGR_H
#define COUNTERMGR_H

#include <fstream>
#include <ostream>
#include <iostream>
#include <dmlc/logging.h>
#include <android/log.h>

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
    __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature", "# release instance #");
    if (instance) {
      instance->model_metric->release_instance();
      instance->model_metric = nullptr;
      instance->msg_publisher->release_instance();
      instance->msg_publisher = nullptr;
      delete instance;
      instance = nullptr;
    }
  }
  static void set_data_consent(int);
  static bool is_feature_enabled();
  bool is_device_info_published() const;
  void runtime_loaded();
  void model_loaded(const std::string& model);
  void model_ran(const std::string& model);
 protected:
  void model_load_publish(record msg_type, const std::string& model);
  void push(string& data) const { 
    if (msg_publisher) {
      __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature", "ccm push =%s", data.c_str());
      msg_publisher->send(data);
    }
  };
 private:
  CounterMgr();
  ~CounterMgr() {
    delete system;
    system = nullptr;
  }
  CounterMgr(const CounterMgr&) {};
  CounterMgr& operator=(const CounterMgr& obj) {return *this;}; 
  // fields for matric data type
  static int feature_enable;
  System *system;
  static CounterMgr* instance;
  MsgPublisher* msg_publisher;
  ModelMetric* model_metric;
};

/*! \brief Hook for Call Home Feature.
 */
extern CounterMgr *instance;
inline void CallHome(record type, std::string model= std::string())
{
  CounterMgr* instance;
  if (!instance) {
    #if defined(__ANDROID__)
    try {
      instance = CounterMgr::get_instance();
      if (!instance)  {
        LOG(FATAL) << "Call Home Feature not initialize!";
      }
    } catch (std::exception& e) {
      instance = nullptr;
      LOG(FATAL) << "Exception in Counter Manger Module initialization";
    }
    #else
    return;
    #endif
  }
  switch (type) {
    case RUNTIME_LOAD:
      instance->runtime_loaded();
      break;
    case MODEL_LOAD:
      instance->model_loaded(model);
      break;
    case MODEL_RUN:
      instance->model_ran(model);
      break;
    default:
      instance->release_instance();
      instance = nullptr;
      break;
  }
}

#endif //COUNTERMGR_H
