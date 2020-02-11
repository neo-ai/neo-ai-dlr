#include <mutex>

#include "counter/counter_mgr.h"
std::mutex g_ccm_mutex;

CounterMgr::CounterMgr() {
  try {
    // Msg Publisher
    msg_publisher = MsgPublisher::get_instance();
    if (!msg_publisher) LOG(FATAL) << "Call Home Message Publisher object null !";
    #if defined(__ANDROID__)
    system = Factory::get_system(ANDROIDS);
    #endif
    if (!system) { 
      LOG(FATAL) << "Call Home feature not supported!"; 
      throw std::runtime_error("Non supported system by call home feature C-API");
    } 
    model_metric = ModelMetric::get_instance();
    if (!model_metric) LOG(FATAL) << "Call Home model metric object null !";
  } catch (std::exception& e) {
    LOG(FATAL) << "Exception in Counter Manger constructor";
    throw e; 
  }
}

CounterMgr* CounterMgr::get_instance() {
  g_ccm_mutex.lock();
  if (!instance) {
    if (is_feature_enabled()) {
      LOG(INFO) << CALL_HOME_USR_NOTIFICATION  << std::endl;
      try {
        instance = new CounterMgr();
      } catch (std::exception& e) {
        instance = nullptr;
      }
      if (instance) {
        instance->runtime_loaded();
      }
    } else LOG(INFO) << "call home feature disabled" << std::endl;
           
  }
  g_ccm_mutex.unlock();
  return instance;
};

void CounterMgr::set_data_consent(int val) {
  feature_enable = val;
}

bool CounterMgr::is_feature_enabled() {
  if (feature_enable)
    return true;
  else return false;
};

bool CounterMgr::is_device_info_published() const {
  #if defined(__ANDROID__)
  std::string file_path(ext_path.c_str());
  file_path += "/";
  file_path += CALL_HOME_RECORD_FILE;
  ifstream fin;
  fin.open(file_path);
  if (fin.is_open()) {
    fin.close();
    return true; 
  } else {
    ofstream fout;
    fout.open(file_path);
    if (fout.is_open()) {
      string id = system->get_device_id();
      fout << id << std::endl;
    }
    fout.close();
    return false;
  }
  #else
  return false;
  #endif
}

void CounterMgr::runtime_loaded() {
  if (!is_device_info_published()) {
    char buff[256];
    snprintf(buff, sizeof(buff), "{ \"record_type\": %s, %s }", std::to_string(RUNTIME_LOAD).c_str(), system->get_device_info().c_str()); 
    std::string str_pub = buff;
    push(str_pub);
  }
};

void CounterMgr::model_load_publish(record msg_type, const std::string& model) {
  char buff[128];
  snprintf(buff, sizeof(buff), "{ \"record_type\": %s, \"model\":\"%s\", \"uuid\": \"%s\" }", std::to_string(msg_type).c_str(), get_hash_string(model).c_str(), system->get_device_id().c_str()); 
  std::string str_pub = buff;
  push(str_pub);
};

void CounterMgr::model_loaded(const std::string& model) {
  std::string uid = instance->system->get_device_id();
  instance->model_metric->set_device_id(uid);
  model_load_publish(MODEL_LOAD, model);
};

void CounterMgr::model_ran(const std::string& model) {
  ModelExecCounter::update_model_run_count(get_hash_string(model).c_str());
}

CounterMgr * CounterMgr::instance = nullptr;
int CounterMgr::feature_enable = 1;
