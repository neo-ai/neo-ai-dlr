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
  virtual ~System() {};
};

class ARM : public System {
};

class Linux_ARM: public ARM {
 public:
  Linux_ARM()
  {
    device = new DeviceInfo();
    #if defined(__LINUX__) ||  defined(__linux__)
    device->osname.assign("Linux ARM");
    device->uuid.assign("1234566");
    device->dist.assign("LINUX Release ARM");
    device->name.assign("LINUX");
    #endif
  }
  ~Linux_ARM()
  {
    delete device;
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
  ~Android()
  {
     delete device;
  }
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
     device = new DeviceInfo();
     #if defined(__LINUX__)
     #endif
  }
  ~Linux_x86()
  {
     delete device;
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