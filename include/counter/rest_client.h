#ifndef RESTCLIENT_H
#define RESTCLIENT_H

#include <iostream>
#include <fstream>
#include <dmlc/logging.h>

#if defined(__ANDROID__)
#include <curl/curl.h>
#endif

#include "config.h"

/*! \brief class RestClient 
 */
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
    CURL *curl;
    CURLcode res;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if(curl) {
      struct curl_slist *headers=NULL;
      headers=curl_slist_append(headers, "Content-Type: application/x-amz-json-1.1; charset=UTF-8");
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
      curl_easy_setopt(curl, CURLOPT_URL, CALL_HOME_URL.c_str());
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
      char *s = curl_easy_escape(curl, data.c_str(), data.length());       
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, s);
      res = curl_easy_perform(curl);
      if(res != CURLE_OK) {
        LOG(INFO) << "Rest client perform return code :" << res; 
      } else {
        long resp_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp_code);
        LOG(INFO) << "Rest client response code :" << resp_code;
      } 
      curl_free(s);
      curl_easy_cleanup(curl);
      curl_slist_free_all(headers);
    }
    curl_global_cleanup();
    #endif
  };
 private:
};

#endif //RESTCLIENT_H
