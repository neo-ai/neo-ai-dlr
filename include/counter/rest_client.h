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
  #if defined(__ANDROID__)
  #endif
  RestClient() {
     #if defined(__ANDROID__)
     #endif
  };
  ~RestClient() {
     #if defined(__ANDROID__)
     #endif
  };
  void send(std::string data) {
    std::cout << "Sending message " << data << std::endl; 
     #if defined(__ANDROID__)
    /* 
    CURL *curl;
    CURLcode res;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    #if defined(__ANDROID__)
      //__android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "inside restclient send %s ", data.c_str());
    #endif
    if(curl) {

      struct curl_slist *headers=NULL;
      headers=curl_slist_append(headers, "Content-Type: application/x-amz-json-1.1");
      curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:5000/dlr_device_info");
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
      res = curl_easy_perform(curl);
      if(res != CURLE_OK)
        fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));
      curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
    */
     #endif
  };
 private:
   
};

#endif //MY_APPLICATION_RESTCLIENT_H
