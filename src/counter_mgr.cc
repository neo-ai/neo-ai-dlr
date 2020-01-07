#include "counter/counter_mgr.h"

CounterMgr::CounterMgr()
{
  // Msg Publisher
  msg_publisher = MsgPublisher::get_instance();
  #if defined(__ANDROID__)
  system = Factory::get_system(3);
  #elif defined(__LINUX__) || defined(__linux__)
  system = Factory::get_system(1);
  #endif
  model_metric = ModelMetric::get_instance();
}

CounterMgr* CounterMgr::get_instance()
{
  if (is_feature_enabled()) {
    if (!instance) {
      instance = new CounterMgr();
    }
  }
  return instance;
};

bool CounterMgr::is_feature_enabled()
{
  /*std::string file_path = "/sdcard/";
  file_path += CALL_HOME_USER_CONFIG_FILE;
  ifstream fin(file_path);
  if (fin.is_open())
  {
    return true;
  }
  else {
    ofstream fout(file_path);
    fout << 1 << endl;
    fout.close();
  } */ 
  return true;
};

bool CounterMgr::is_device_info_published()
{
  /*std::string file_path = "/sdcard/";
  file_path += CALL_HOME_RECORD_FILE;

  ifstream fin;
  fin.open(file_path);
  std::string dev_id;
  getline(fin, dev_id);
  //std::string str (id);

  if (dev_id.length() < 2)
  {
    ofstream fout;
    fout.open(file_path);
    std::string id = system->retrieve_id();
    system->set_device_id(id);
    fout << id << std::endl;
    fout.close();
    return false;
  }
  system->set_device_id(dev_id);
  fin.close();*/
  return false;
}

void CounterMgr::runtime_loaded()
{
  if (!is_device_info_published())
  {
    std::string pubdata = "{\"record_type\":";
    pubdata += std::to_string(RUNTIME_LOAD) + ", ";
    pubdata += system->get_device_info();
    pubdata += "}";
    push(pubdata);
  }
};

void CounterMgr::model_info_published(int msg_type, std::string model, int count)
{
  std::string str_pub = "{\"record_type\":";
  str_pub += std::to_string(msg_type) + ", ";
  str_pub += "\"model\":\"";
  std::size_t model_hash = std::hash<std::string>{}(model);
  str_pub += std::to_string(model_hash) + "\", ";
  str_pub += "\"uuid\":";
  str_pub += "\"" + system->get_device_id() + "\" }";
  push(str_pub);
};

void CounterMgr::model_loaded(std::string model) {
  std::string uid = instance->system->get_device_id();
  instance->model_metric->set_device_id(uid);
  model_info_published(MODEL_LOAD, model);
};

void CounterMgr::model_run(std::string model) {
  std::size_t model_hash = std::hash<std::string>{}(model);
  ModelExecCounter::add_model_run_count(model_hash);
}

CounterMgr * CounterMgr::instance = nullptr;
