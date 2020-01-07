#ifndef MY_APPLICATION_MODEL_METRIC_H
#define MY_APPLICATION_MODEL_METRIC_H

#include <thread>
#include <queue>
#include <fstream>
#include <ostream>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include "rest_client.h"
#include "config.h"
#include "model_exec_counter.h"

class ModelMetric {
 public:
  static ModelMetric* modelmetric;
  static ModelMetric* get_instance();
  static void release_instance() {
    modelmetric->stop_process = true;
    modelmetric->thrd->join();
    delete modelmetric;
    modelmetric = nullptr;
  }
  ~ModelMetric();
  void process_queue();
  void set_device_id(std::string dev_id) {
    device_id.assign(dev_id);
  }
 private:
  ModelMetric() {
    restcon = new RestClient();
    stop_process = false;
  };
  RestClient *restcon;
  std::thread *thrd;
  bool stop_process;
  std::string device_id;
  const int MODEL_RUN = 3;
};

#endif //MY_APPLICATION_MODEL_METRIC_H
