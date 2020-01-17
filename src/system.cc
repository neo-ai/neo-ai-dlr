#include "counter/system.h"
#include <android/log.h>
Android::Android()
{
  #if defined(__ANDROID__)
  device = new DeviceInfo();
  char value[PROP_VALUE_MAX+1];
  int osVersionLength = __system_property_get("ro.build.version.release", value);
  device->dist.assign(value);
  string os_name ("Android ");
  os_name += value;
  device->osname.assign(os_name);
  __system_property_get("ro.product.cpu.abi", value);
  device->arch.assign(value);
  __system_property_get("ro.build.host", value);
  device->machine.assign(value);
  __system_property_get("ro.product.name", value);
  device->name.assign(value);
  device->uuid.assign(uuid_.c_str());
  #endif
};

