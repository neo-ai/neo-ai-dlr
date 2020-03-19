#ifndef COUNTERMGR_H
#define COUNTERMGR_H

#include <fstream>
#include <ostream>
#include <iostream>
#include <dmlc/logging.h>
#include <thread>
#include <deque>
#include <map>

#include "device_info.h"
#include "system.h"
#include "config.h"
#include "helper.h"
#include "rest_client.h"

extern std::string ext_path;

enum record { RUNTIME_LOAD=1, MODEL_LOAD=2, MODEL_RUN=3, CM_RELEASE=9};

/*! \brief class CounterMgr
 */
class CounterMgr {
 public:
  static CounterMgr* get_instance();
  static void release_instance() {
    if (instance) {
      instance->stop_process = true;
      instance->thrd->join();
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
  void process_queue();
  void publish_msg();
 protected:
  void model_load_publish(record msg_type, const std::string& model);
  void push(std::string data) {
    msg_que.push_back(data); 
  };
 private:
  CounterMgr();
  ~CounterMgr() {
    publish_msg();
    delete thrd;
    thrd = nullptr;
    delete restcon;
    restcon = nullptr;
    delete system;
    system = nullptr;
  }
  CounterMgr(const CounterMgr&) {};
  CounterMgr& operator=(const CounterMgr& obj) {return *this;}; 
  static bool feature_enable;
  System *system;
  static CounterMgr* instance;
  std::map<std::string, int > model_dict;
  RestClient *restcon;
  std::thread *thrd;
  std::deque<std::string> msg_que;
  bool stop_process;
  int retrycnt;
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
