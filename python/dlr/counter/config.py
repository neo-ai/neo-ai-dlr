call_home_url = 'https://r4sir2jo37.execute-api.us-east-2.amazonaws.com/testing/'
call_home_record_file = 'ccm_record.txt'
call_home_user_config_file = 'ccm_config.json'
call_home_publish_message_max_queue_size = 100
call_home_max_workers_threads = 5
call_home_model_run_count_time = 5
call_home_usr_notification = """\n Your device information and Model metric data being sent to a server. \
                            \n You can disable this feature by using below configuration steps. \
                            \n\t1. Create a config file, ccm_config.json in your application path. \
                            \n\t2. Added below format content in it, \
                            \n\t\t{\n\t\t\t"ccm" : "false"\n\t\t}"""
