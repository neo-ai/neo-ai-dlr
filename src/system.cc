#include "counter/system.h"
Android::Android() {
  #if defined(__ANDROID__)
  device_ = new DeviceInfo();
  char value[PROP_VALUE_MAX+1];
  int osVersionLength = __system_property_get("ro.build.version.release", value);
  device_->dist.assign(value);
  std::string os_name ("Android ");
  os_name += value;
  device_->osname.assign(os_name);
  __system_property_get("ro.product.cpu.abi", value);
  device_->arch.assign(value);
  __system_property_get("ro.build.host", value);
  device_->machine.assign(value);
  __system_property_get("ro.product.name", value);
  device_->name.assign(value);
  device_->uuid.assign(uuid_.c_str());
  #endif
};

