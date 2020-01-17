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
extern std::string uuid_;
using namespace std;

/*! \brief class System (Abstract Class) 
 */
class System {
 public:
  virtual std::string get_device_info() const =0;
  virtual std::string get_device_id() const =0;
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
 private:
  DeviceInfo* device;
};

/*! \brief class Android
 */
class Android : public ARM
{
 public:
  Android();
  ~Android()
  {
     delete device;
  }
  std::string get_device_info() const
  {
    return device->get_info();
  };
  std::string get_device_id() const
  {
    return device->uuid;
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
