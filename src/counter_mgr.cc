#include "counter/counter_mgr.h"

CounterMgr::CounterMgr() {
  try {
    #if defined(__ANDROID__)
    system_ = Factory::GetSystem(ANDROIDS);
    #endif
    if (!system_) { 
      LOG(FATAL) << "Call Home feature not supported!"; 
      throw std::runtime_error("Non supported system by call home feature C-API");
    } 
    restcon_ = new RestClient();
    if (!restcon_) {
      stop_process_ = true;
      LOG(FATAL) << "Message Publisher object null !";
      throw std::runtime_error("Message Publisher object null !");
    }
    stop_process_ = false;
    thrd_ = nullptr;
    retrycnt_ = 0;
  } catch (std::exception& e) {
    LOG(FATAL) << "Exception in Counter Manger constructor";
    throw e; 
  }
}

CounterMgr* CounterMgr::GetInstance() {
  if (!instance_) {
    if (GetFeatureEnabled()) {
      LOG(INFO) << CALL_HOME_USR_NOTIFICATION  << std::endl;
      try {
        instance_ = new CounterMgr();
        if (instance_) {
           instance_->thrd_ = new std::thread(&CounterMgr::ProcessQueue, instance_);
        }
      } catch (std::exception& e) {
        instance_ = nullptr;
      }
      if (instance_) {
        instance_->RuntimeLoaded();
      }
    } else LOG(INFO) << "call home feature disabled" << std::endl;
           
  }
  return instance_;
};

void CounterMgr::SetDataConsent(int val) {
  feature_enable_ = val;
}

bool CounterMgr::GetFeatureEnabled() {
  return feature_enable_;
};

bool CounterMgr::GetDeviceInfoPublished() const {
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
      std::string id = system_->GetDeviceId();
      fout << id << std::endl;
    }
    fout.close();
    return false;
  }
  #else
  return false;
  #endif
}

void CounterMgr::RuntimeLoaded() {
  if (!GetDeviceInfoPublished()) {
    msg_que_.push_back(runtime_load_);
  }
};

void CounterMgr::ModelLoaded(const std::string& model) {
  std::string str(model_load_);
  str += GetHashString(model).c_str();
  msg_que_.push_back(str);
};

void CounterMgr::ModelRan(const std::string& model) {
  msg_que_.push_back(model);
};

void CounterMgr::ProcessQueue() {
  while (!stop_process_) {
    std::unique_lock<std::mutex> lk(condv_m_);
    condv_.wait_for(lk, std::chrono::seconds(CALL_HOME_MODEL_RUN_COUNT_TIME_SECS));
    PublishMsg();
  }
};

void CounterMgr::PublishMsg() {
  std::map<std::string, int> model_dict;
  std::string uuid(system_->GetDeviceId());

  while (!msg_que_.empty()) {
    std::string msg_typ = msg_que_.front();
    if (msg_typ.compare(runtime_load_) == 0) {
      char buff[256];
      snprintf(buff, sizeof(buff), "{ \"record_type\": %d, %s }", RUNTIME_LOAD, system_->GetDeviceInfo().c_str());
      std::string msg(buff);
      SendMsg(msg); 
    } else if (msg_typ.find(model_load_) != std::string::npos) {
      char buff[128];
      std::string model = msg_typ.substr(msg_typ.find(model_load_)+model_load_.length()); 
      snprintf(buff, sizeof(buff), "{ \"record_type\": %d, \"model\": \"%s\", \"uuid\": \"%s\" }", 
               MODEL_LOAD, GetHashString(model).c_str(), uuid.c_str());
      std::string msg(buff);
      SendMsg(msg); 
    } else {
      model_dict[msg_typ] += 1;
    }
    msg_que_.pop_front();
  }

  for (auto pair_dict : model_dict) {
    char buff[135];
    snprintf(buff, sizeof(buff), "{ \"record_type\": %d, \"model\": \"%s\", \"uuid\": \"%s\", \"run_count\": %d }", 
             MODEL_RUN, GetHashString(pair_dict.first).c_str(), uuid.c_str(), pair_dict.second);
    std::string msg(buff);
    SendMsg(msg);
  }
};

void CounterMgr::SendMsg(const std::string& msg)
{
  if (retrycnt_ < CALL_HOME_REQ_STOP_MAX_COUNT) {
    int status = restcon_->Send(msg);
    if (status != 200) retrycnt_++;
  }
}

CounterMgr * CounterMgr::instance_ = nullptr;
bool CounterMgr::feature_enable_ = true;
