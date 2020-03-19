#include "counter/counter_mgr.h"

CounterMgr::CounterMgr() {
  try {
    #if defined(__ANDROID__)
    system = Factory::get_system(ANDROIDS);
    #endif
    if (!system) { 
      LOG(FATAL) << "Call Home feature not supported!"; 
      throw std::runtime_error("Non supported system by call home feature C-API");
    } 
    restcon = new RestClient();
    if (!restcon) {
      stop_process = true;
      LOG(FATAL) << "Message Publisher object null !";
      throw std::runtime_error("Message Publisher object null !");
    }
    stop_process = false;
    thrd = nullptr;
    retrycnt = 0;
  } catch (std::exception& e) {
    LOG(FATAL) << "Exception in Counter Manger constructor";
    throw e; 
  }
}

CounterMgr* CounterMgr::get_instance() {
  if (!instance) {
    if (is_feature_enabled()) {
      LOG(INFO) << CALL_HOME_USR_NOTIFICATION  << std::endl;
      try {
        instance = new CounterMgr();
        if (instance) {
           instance->thrd = new std::thread(&CounterMgr::process_queue, instance);
        }
      } catch (std::exception& e) {
        instance = nullptr;
      }
      if (instance) {
        instance->runtime_loaded();
      }
    } else LOG(INFO) << "call home feature disabled" << std::endl;
           
  }
  return instance;
};

void CounterMgr::set_data_consent(int val) {
  feature_enable = val;
}

bool CounterMgr::is_feature_enabled() {
  return feature_enable;
};

bool CounterMgr::is_device_info_published() const {
  #if defined(__ANDROID__)
  std::string file_path(ext_path.c_str());
  file_path += "/";
  file_path += CALL_HOME_RECORD_FILE;
  std::ifstream fin;
  fin.open(file_path);
  if (fin.is_open()) {
    fin.close();
    return true; 
  } else {
    std::ofstream fout;
    fout.open(file_path);
    if (fout.is_open()) {
      std::string id = system->get_device_id();
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
    snprintf(buff, sizeof(buff), "{ \"record_type\": %d, %s }", RUNTIME_LOAD, system->get_device_info().c_str());
    std::string str_pub = buff;
    push(str_pub);
  }
};

void CounterMgr::model_loaded(const std::string& model) {
  char buff[128];
  snprintf(buff, sizeof(buff), "{ \"record_type\": %d, \"model\": \"%s\", \"uuid\": \"%s\" }", MODEL_LOAD, get_hash_string(model).c_str(), system->get_device_id().c_str());
  std::string str_pub(buff);
  push(str_pub);
};

void CounterMgr::model_ran(const std::string& model) {
  run_deq.push_back(model);
};

void CounterMgr::process_queue() {
  while (!stop_process) {
    std::unique_lock<std::mutex> lk(condv_m);
    condv.wait_for(lk, std::chrono::seconds(CALL_HOME_MODEL_RUN_COUNT_TIME_SECS));
    publish_msg();
  }
};

void CounterMgr::publish_msg() {
  std::map<std::string, int> model_dict;
  std::string uuid(system->get_device_id());

  while (!run_deq.empty()) {
    model_dict[run_deq.front()] += 1;
    run_deq.pop_front();
  }

  for (auto pair_dict : model_dict) {
    char buff[135];
    snprintf(buff, sizeof(buff), "{ \"record_type\": %d, \"model\": \"%s\", \"uuid\": \"%s\", \"run_count\": %d }", MODEL_RUN, get_hash_string(pair_dict.first).c_str(), uuid.c_str(), pair_dict.second);
    std::string pub_data(buff);
    push(pub_data);
  }

  while (!msg_que.empty()) {
    std::string msg = msg_que.front();
    if (retrycnt < CALL_HOME_REQ_STOP_MAX_COUNT) {
      int status = restcon->send(msg);
      if (status != 200) retrycnt++;
    }
    msg_que.pop_front();
  }
};

CounterMgr * CounterMgr::instance = nullptr;
bool CounterMgr::feature_enable = true;
