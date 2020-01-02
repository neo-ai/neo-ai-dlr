#include "counter/system.h"

Android::Android()
{
  #if defined(__ANDROID__)
  device = new DeviceInfo();
  // os version, dist
  char value[PROP_VALUE_MAX+1];
  int osVersionLength = __system_property_get("ro.build.version.release", value);
  device->dist.assign(value);
  // os name
  string os_name ("Android ");
  os_name += value;
  device->osname.assign(os_name);
  // cpu arch
  __system_property_get("ro.product.cpu.abi", value);
  device->arch.assign(value);
  __system_property_get("ro.build.host", value);
  device->machine.assign(value);
  // machine name
  __system_property_get("ro.product.name", value);
  device->name.assign(value);
  #endif
};

std::string Android::retrieve_id()
{
  random_device rd;
  mt19937 gen(rd());
  std::stringstream stream;
  stream << std::hex << gen();
  stream << std::hex << gen();
  device->uuid = stream.str();
  return device->uuid;
};
