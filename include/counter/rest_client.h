#ifndef MY_APPLICATION_RESTCLIENT_H
#define MY_APPLICATION_RESTCLIENT_H

#include <iostream>
#include <fstream>

#if defined(__ANDROID__)
#include <android/log.h>
#include <curl/curl.h>
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
    #if defined(__ANDROID__)
    __android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "inside restclient send %s ", data.c_str());
     
    CURL *curl;
    CURLcode res;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    std::string readBuffer;
 
    if(curl) {

      struct curl_slist *headers=NULL;
      headers=curl_slist_append(headers, "Content-Type: application/x-amz-json-1.1; charset=UTF-8");
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
 
      curl_easy_setopt(curl, CURLOPT_URL, CALL_HOME_URL.c_str());
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

      char *s = curl_easy_escape(curl, data.c_str(), data.length());       
      __android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "New string =%s ", s);
     
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, s);

      res = curl_easy_perform(curl);
        __android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "restclient curl perform res =%d ", res);
     
       
      if(res != CURLE_OK) {
        __android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "Error restclient curl perform ");
        fprintf(stderr, "curl_easy_perform() failed: %s\n",
                curl_easy_strerror(res));
        __android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "Error Msg =%s ",curl_easy_strerror(res) );
      } else {
        long resp_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp_code);
        __android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "Resp CODE =%ld ",resp_code );
        
      } 

      curl_easy_cleanup(curl);
      curl_slist_free_all(headers);
    }

     
    curl_global_cleanup();
    
    #endif
  };
 private:
   
};

#endif //MY_APPLICATION_RESTCLIENT_H
