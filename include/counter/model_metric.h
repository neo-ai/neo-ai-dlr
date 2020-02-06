#ifndef MODEL_METRIC_H
#define MODEL_METRIC_H

#include <thread>
#include <queue>
#include <fstream>
#include <ostream>

#include "rest_client.h"
#include "config.h"
#include "model_exec_counter.h"
#include "publisher.h"

/*! \brief class ModelMetric 
 */
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
    publisher = MsgPublisher::get_instance();
    stop_process = false;
  };
  MsgPublisher *publisher; 
  std::thread *thrd;
  bool stop_process;
  std::string device_id;
  const int MODEL_RUN = 3;
};

#endif //MODEL_METRIC_H
