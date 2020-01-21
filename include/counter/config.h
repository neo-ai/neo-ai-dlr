#ifndef CONFIG_H
#define CONFIG_H

const std::string CALL_HOME_URL = "https://r4sir2jo37.execute-api.us-east-2.amazonaws.com/testing/";
const std::string CALL_HOME_RECORD_FILE = "ccm_record.txt";
const std::string CALL_HOME_USER_CONFIG_FILE = "ccm_config.json";
const int CALL_HOME_PUBLISH_MESSAGE_MAX_QUEUE_SIZE = 100;
const int CALL_HOME_MAX_WORKERS_THREADS = 5;
const int CALL_HOME_MODEL_RUN_COUNT_TIME_SECS = 300;
const std::string CALL_HOME_USR_NOTIFICATION = "\n CALL HOME FEATURE ENABLED" \
        "\n Your device information and model load/execution count metric data being sent to a server." \
        "\n You can disable this feature by using below configuration steps." \
        "\n\t 1. Call DLR C-API SetDataCollectionConsent(0)" \
        "\n\t Params, 0 - feature disable, 1 - feature enable";

#endif //CONFIG_H
