#ifndef MY_APPLICATION_CONFIG_H
#define MY_APPLICATION_CONFIG_H

const std::string CALL_HOME_URL = "https://r4sir2jo37.execute-api.us-east-2.amazonaws.com/testing/";
const std::string CALL_HOME_RECORD_FILE = "ccm_record.txt";
const std::string CALL_HOME_USER_CONFIG_FILE = "ccm_config.json";
const int CALL_HOME_PUBLISH_MESSAGE_MAX_QUEUE_SIZE = 100;
const int CALL_HOME_MAX_WORKERS_THREADS = 5;
const int CALL_HOME_MODEL_RUN_COUNT_TIME_SECS = 5;
const std::string CALL_HOME_USR_NOTIFICATION = "\n CALL HOME FEATURE DISABLED" \
        "\n Your device information and model load/execution count metric data being sent to a server." \
        "\n You can disable this feature by using below configuration steps." \
        "\n\t 1. Create a config file, ccm_config.json in your application external storage path." \
        "\n\t2. Added below format content in it," \
        "\n\t\t{\n\t\t\t ccm : false \n\t\t}";

#endif //MY_APPLICATION_CONFIG_H
