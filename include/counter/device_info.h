#ifndef MY_APPLICATION_DEVICEINFO_H
#define MY_APPLICATION_DEVICEINFO_H

#include <iostream>

using namespace std;

/*! \brief class DeviceInfo, holds device specific information
 */
class DeviceInfo {
 public:
  std::string machine = "";
  std::string arch = "";
  std::string osname = "";
  std::string name = "";
  std::string dist = "";
  std::string uuid = "";

  std::string get_info() const
  {
    std::string str = "\"os distribution\":";
    str += "\"" + dist + "\"";
    str += "\"uuid\":" ;
    str += "\"" + uuid + "\"" ;
    str += "\"machine\":";
    str += "\"" + machine + "\"";
    str += "\"arch\":";
    str += "\"" + arch + "\"";
    str += "\"os\":";
    str += "\"" + osname + "\"";
    str += "\"device\":";
    str += "\"" + name + "\"";
    return str;
  }
};


#endif //MY_APPLICATION_DEVICEINFO_H
