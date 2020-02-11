#ifndef DEVICEINFO_H
#define DEVICEINFO_H

#include <iostream>

using namespace std;

/*! \brief class DeviceInfo, holds device specific information
 */
class DeviceInfo {
 public:
  std::string machine;
  std::string arch;
  std::string osname;
  std::string name;
  std::string dist;
  std::string uuid;

  std::string get_info() const
  {
    char buff[256];
    snprintf(buff, sizeof(buff), " \"os distribution\": \"%s\", \"uuid\": \"%s\", \"machine\": \"%s\", \"arch\": \"%s\", \"os\": \"%s\", \"device\": \"%s\"", dist.c_str(), uuid.c_str(), machine.c_str(), arch.c_str(), osname.c_str(), name.c_str());
    std::string str = buff;
    return str;
  }
};


#endif //DEVICEINFO_H
