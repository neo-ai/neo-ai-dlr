"""configuration keys related to Call Home Feature"""
CALL_HOME_URL = "https://api.neo-dlr.amazonaws.com"
CALL_HOME_USER_CONFIG_FILE = "ccm_config.json"
CALL_HOME_MODEL_RUN_COUNT_TIME_SECS = 300
CALL_HOME_REQ_STOP_MAX_COUNT = 3
CALL_HOME_USR_NOTIFICATION = """\n CALL HOME FEATURE ENABLED
                            \n\n You acknowledge and agree that DLR collects the following metrics to help improve its performance. \
                            \n By default, Amazon will collect and store the following information from your device: \
                            \n\n record_type: <enum, internal record status, such as model_loaded, model_>, \
                            \n arch: <string, platform architecture, eg 64bit>, \
                            \n osname: <string, platform os name, eg. Linux>, \
                            \n uuid: <string, one-way non-identifable hashed mac address, eg. 8fb35b79f7c7aa2f86afbcb231b1ba6e>, \
                            \n dist: <string, distribution of os, eg. Ubuntu 16.04 xenial>, \
                            \n machine: <string, retuns the machine type, eg. x86_64 or i386>, \
                            \n model: <string, one-way non-identifable hashed model name, eg. 36f613e00f707dbe53a64b1d9625ae7d> \
                            \n\n If you wish to opt-out of this data collection feature, please follow the steps below: \
                            \n\t1. Disable it with through code: \
                            \n\t\t from dlr.counter.phone_home import PhoneHome \
                            \n\t\t PhoneHome.disable_feature()\
                            \n\t2. Or, create a config file, ccm_config.json inside your DLR target directory path, i.e. python3.6/site-packages/dlr/counter/ccm_config.json. Then added below format content in it, {"enable_phone_home" : false} \
                            \n\t3. Restart DLR application. \
                            \n\t4. Validate this feature is disabled by verifying this notification is no longer displayed, or programmatically with following command: \
                            \n\t\tfrom dlr.counter.phone_home import PhoneHome \
                            \n\t\tPhoneHome.is_enabled() # false as disabled """