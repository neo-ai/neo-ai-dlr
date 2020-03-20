#ifndef SYSTEM_H
#define SYSTEM_H

#include <iostream>
#include <sstream>

#if defined(__ANDROID__)
  #include <sys/system_properties.h>
  #include <jni.h>
#endif

#include "device_info.h"
extern std::string uuid_;

/*! \brief class System (Abstract Class) 
 */
class System {
 public:
  virtual std::string GetDeviceInfo() const =0;
  virtual std::string GetDeviceId() const =0;
  virtual ~System() {};
};

class ARM : public System {
};

class Linux_ARM: public ARM {
 public:
  Linux_ARM()
  {
    device_ = new DeviceInfo();
    #if defined(__LINUX__) ||  defined(__linux__)
    #endif
  }
  ~Linux_ARM()
  {
    delete device_;
  }
  std::string GetDeviceInfo() const
  {
    return device_->GetInfo();
  }
  std::string GetDeviceId() const
  {
    return device_->uuid;
  }
 private:
  DeviceInfo* device_;
};

/*! \brief class Android
 */
class Android : public ARM
{
 public:
  Android();
  ~Android()
  {
     delete device_;
  }
  std::string GetDeviceInfo() const
  {
    return device_->GetInfo();
  };
  std::string GetDeviceId() const
  {
    return device_->uuid;
  };
 private:
  DeviceInfo* device_;
};

class X86: public System
{};

class Linux_x86:public X86
{
 public:
  Linux_x86()
  {
     device_ = new DeviceInfo();
     #if defined(__LINUX__)
     #endif
  }
  ~Linux_x86()
  {
     delete device_;
  }
  std::string GetDeviceInfo() const
  {
    return device_->GetInfo();
  }
  std::string GetDeviceId() const
  {
    return device_->uuid;
  }
 private:
  DeviceInfo* device_;
};

enum supsys { LINUXARM=1, LINUXX86=2, ANDROIDS=3 };

// get system class instance
class Factory
{
 public:
  static System* GetSystem(supsys choice) {
    if (choice == LINUXARM) {
      return new Linux_ARM();
    } else if (choice == LINUXX86) {
      return new Linux_x86();
    } else if (choice == ANDROIDS) {
      return new Android();
    } else {
      return nullptr;
    }
  }
};

#endif //SYSTEM_H
