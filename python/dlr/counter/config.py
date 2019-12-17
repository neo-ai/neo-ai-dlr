CALL_HOME_URL = 'https://r4sir2jo37.execute-api.us-east-2.amazonaws.com/testing/'
CALL_HOME_RECORD_FILE = 'ccm_record.txt'
CALL_HOME_USER_CONFIG_FILE = 'ccm_config.json'
CALL_HOME_PUBLISH_MESSAGE_MAX_QUEUE_SIZE = 100
CALL_HOME_MAX_WORKERS_THREADS = 5
CALL_HOME_MODEL_RUN_COUNT_TIME_SECS = 5
CALL_HOME_USR_NOTIFICATION = """\n CALL HOME FEATURE DISABLED 
                            \n Your device information and model load/execution count metric data being sent to a server. \
                            \n You can disable this feature by using below configuration steps. \
                            \n\t1. Create a config file, ccm_config.json in your application path. \
                            \n\t2. Added below format content in it, \
                            \n\t\t{\n\t\t\t"ccm" : "false"\n\t\t}"""
