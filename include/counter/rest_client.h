#ifndef MY_APPLICATION_RESTCLIENT_H
#define MY_APPLICATION_RESTCLIENT_H

#include <iostream>
#include <fstream>

#if defined(__ANDROID__)
#include <android/log.h>
//#include <curl/curl.h>
#endif

#include "config.h"

class RestClient {
 public:
  RestClient() {
  };
  ~RestClient() {
  };
  void send(std::string data) {
    #if defined(__ANDROID__)
      //__android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "inside restclient send %s ", data.c_str());
    #endif
    //CURLcode res;
    //CURL *handle;
    //struct curl_slist *headers=NULL;
    // TODO - Convert to reqd. content type
    //'Content-Type': 'application/x-amz-json-1.1'
    //headers = curl_slist_append(headers, "Content-Type: text/xml");
    //handle = curl_easy_init();
    // {
    //      curl_easy_setopt(handle, CURLOPT_POSTFIELDS, data);
    //      curl_easy_setopt(handle, CURLOPT_URL, CALL_HOME_URL.c_str());
    //      curl_easy_perform(handle); /* post away! */
    // }
    //curl_slist_free_all(headers); /* free the header list */
  };
};

#endif //MY_APPLICATION_RESTCLIENT_H
