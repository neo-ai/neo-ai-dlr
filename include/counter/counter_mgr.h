#ifndef MY_APPLICATION_COUNTERMGR_H
#define MY_APPLICATION_COUNTERMGR_H

#include <fstream>
#include <ostream>
#include <iostream>

#include "device_info.h"
#include "system.h"
#include "publisher.h"
#include "config.h"
#include "model_exec_counter.h"
#include "model_metric.h"

using namespace std;
 
extern const char* ext_path;

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
  System *system;
  static CounterMgr* instance;
  MsgPublisher* msg_publisher;
  ModelMetric* model_metric;
};


#endif //MY_APPLICATION_COUNTERMGR_H
