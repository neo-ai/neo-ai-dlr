"""configuration keys related to Call Home Feature"""
CALL_HOME_URL = 'https://api.neo-dlr.amazonaws.com'
CALL_HOME_RECORD_FILE = 'ccm_record.txt'
CALL_HOME_USER_CONFIG_FILE = 'ccm_config.json'
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
                            \n\t1. Create a config file, ccm_config.json in your application path. \
                            \n\t2. Added below format content in it, \
                            \n\t\t{\n\t\t\t"ccm" : "false"\n\t\t} \
                            \n\t3. Restart DLR application."""
