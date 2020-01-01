#ifndef MY_APPLICATION_SYSTEM_H
#define MY_APPLICATION_SYSTEM_H

#include <iostream>
#include <random>
#include <sstream>

#if defined(__ANDROID__)
  #include <sys/system_properties.h>
  #include <jni.h>
#endif

#include "device_info.h"
using namespace std;

class System {
 public:
  virtual std::string get_device_info() const =0;
  virtual std::string get_device_id() const =0;
  virtual void set_device_id(std::string str) =0;
  virtual std::string retrieve_id() =0;
};

class ARM : public System {
};

class Linux_ARM: public ARM {
 public:
  Linux_ARM()
  {
        #ifdef LINUX

        #endif
  }
  std::string get_device_info() const
  {
    return device->get_info();
  }
  std::string get_device_id() const
  {
    return device->uuid;
  }
  void set_device_id(string id)
  {
    device->uuid.assign(id);
  }
  std::string retrieve_id()
  {
    return "";
  };
 private:
  DeviceInfo* device;
};

class Android : public ARM
{
 public:
  Android();
  std::string retrieve_id();
  std::string get_device_info() const
  {
    return device->get_info();
  };
  std::string get_device_id() const
  {
    return device->uuid;
  };
  void set_device_id(std::string id)
  {
    device->uuid.assign(id);
  };
 private:
  DeviceInfo* device;
};

class X86: public System
{};

class Linux_x86:public X86
{
 public:
  Linux_x86()
  {
        #ifdef LINUX_X86
        #endif
  }
  std::string get_device_info() const
  {
    return device->get_info();
  }
  std::string get_device_id() const
  {
    return device->uuid;
  }
  void set_device_id(std::string str)
  {
    device->uuid.assign(str);
  }
  std::string retrieve_id() {
    return "";
  }
 private:
  DeviceInfo* device;
};

// get system class instance
class Factory
{
 public:
  static System* get_system(int choice) {
    if (choice == 1) {
      return new Linux_ARM();
    } else if (choice == 2) {
      return new Linux_x86();
    } else if (choice == 3) {
      return new Android();
    } else {
      return nullptr;
    }
  }
};

#endif //MY_APPLICATION_SYSTEM_H
