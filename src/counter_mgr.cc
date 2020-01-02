#include "counter/counter_mgr.h"

CounterMgr::CounterMgr()
{
  // Msg Publisher
  msg_publisher = MsgPublisher::get_instance();

  //TODO - Dynamic retrieve system
  system = Factory::get_system(3);

  //
}

CounterMgr* CounterMgr::get_instance()
{
  if (is_feature_enabled()) {
    if (instance)
      return instance;
    else {
      instance = new CounterMgr();
    }
  }
  return instance;
};

bool CounterMgr::is_feature_enabled()
{
  std::string file_path = "/sdcard/";
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
  }
  return true;
};

bool CounterMgr::is_device_info_published()
{
  std::string file_path = "/sdcard/";
  file_path += CALL_HOME_RECORD_FILE;

  ifstream fin;
  fin.open(file_path);

  test_str += "\n file open for reading, ";
  std::string id;
  getline(fin, id);
  std::string str (id);

  if (id.length() < 2)
  {
    test_str += "\n ELSE file open for writing, ";
    ofstream fout;
    fout.open(file_path);
    //if (fout.is_open()){
    test_str += "\n file open for writing, ";
    std::string id = system->retrieve_id();
    test_str += "id =>";
    test_str += id;
    system->set_device_id(id);

    fout << id << std::endl;
    fout.close();
    return false;
  }
  test_str += str;
  system->set_device_id(str);
  fin.close();
  return true;
}

void CounterMgr::runtime_loaded()
{
  if (!is_device_info_published())
  {
    std::string pubdata = "record_type:";
    pubdata += std::to_string(RUNTIME_LOAD);
    pubdata += system->get_device_info();

    test_str += pubdata;
  }
  else
    test_str += "Device Information already published !";
  //test_str += pSystem->get_device_info();
};


void CounterMgr::model_info_published(int msg_type, std::string model, int count)
{
  std::string str_pub = "{\"record_type\":";
  str_pub += std::to_string(msg_type) + ", ";
  str_pub += "\"model\":";
  str_pub += model + ", ";
  str_pub += "\"uuid\":";
  str_pub += "\"" + system->get_device_id() + "\" }";

  test_str += str_pub;
  push(str_pub);
};

CounterMgr * CounterMgr::instance = nullptr;
