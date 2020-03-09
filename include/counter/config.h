#ifndef CONFIG_H
#define CONFIG_H

const std::string CALL_HOME_URL = "https://beta-neo-dlr.us-west-2.amazonaws.com";
const std::string CALL_HOME_RECORD_FILE = "ccm_record.txt";
const int CALL_HOME_MODEL_RUN_COUNT_TIME_SECS = 300;
const int CALL_HOME_REQ_STOP_MAX_COUNT = 3;
const std::string CALL_HOME_USR_NOTIFICATION = "\n CALL HOME FEATURE ENABLED" \
        "\n Your device information and model load/execution count metric data being sent to a server." \
        "\n You can disable this feature by using below configuration steps." \
        "\n\t 1. Call DLR C-API SetDLRDataCollectionConsent(0)" \
        "\n\t Params, 0 - feature disable, 1 - feature enable";

#endif //CONFIG_H
