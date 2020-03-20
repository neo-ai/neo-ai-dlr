#ifndef COUNTERMGR_H
#define COUNTERMGR_H

#include <fstream>
#include <ostream>
#include <iostream>
#include <dmlc/logging.h>
#include <thread>
#include <condition_variable>
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
  static CounterMgr* GetInstance();
  static void ReleaseInstance() {
    if (instance_) {
      instance_->stop_process_ = true;
      instance_->condv_.notify_all();
      instance_->thrd_->join();
      delete instance_;
      instance_ = nullptr;
    }
  }
  static void SetDataConsent(int);
  static bool GetFeatureEnabled();
  bool GetDeviceInfoPublished() const;
  void RuntimeLoaded();
  void ModelLoaded(const std::string& model);
  void ModelRan(const std::string& model);
  void ProcessQueue();
  void PublishMsg();
 protected:
  void Push(std::string data) {
    msg_que_.push_back(data); 
  };
 private:
  CounterMgr();
  ~CounterMgr() {
    delete thrd_;
    thrd_ = nullptr;
    delete restcon_;
    restcon_ = nullptr;
    delete system_;
    system_ = nullptr;
  }
  CounterMgr(const CounterMgr&) {};
  CounterMgr& operator=(const CounterMgr& obj) {return *this;};
  void SendMsg(const std::string& msg); 
  static bool feature_enable_;
  System *system_;
  static CounterMgr* instance_;
  RestClient *restcon_;
  std::thread *thrd_;
  std::deque<std::string> msg_que_;
  bool stop_process_;
  int retrycnt_;
  std::mutex condv_m_;
  std::condition_variable condv_;
  const std::string runtime_load_="runtimeload|";
  const std::string model_load_="modelload|";
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
      instance = CounterMgr::GetInstance();
      if (!instance)  {
        LOG(FATAL) << "Call Home Feature not initialize!";
        return;
      }
    } catch (std::exception& e) {
      instance = nullptr;
      LOG(FATAL) << "Exception in Counter Manger Module initialization";
      return;
    }
    #else
    return;
    #endif
  }
  switch (type) {
    case RUNTIME_LOAD:
      instance->RuntimeLoaded();
      break;
    case MODEL_LOAD:
      instance->ModelLoaded(model);
      break;
    case MODEL_RUN:
      instance->ModelRan(model);
      break;
    default:
      instance->ReleaseInstance();
      instance = nullptr;
      break;
  }
}

#endif //COUNTERMGR_H
