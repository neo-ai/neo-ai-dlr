#ifndef RESTCLIENT_H
#define RESTCLIENT_H

#include <iostream>
#include <fstream>
#include <dmlc/logging.h>

#if defined(__ANDROID__)
#include <curl/curl.h>
#include <android/log.h>
#endif

#include "config.h"
static size_t WriteCallback(char *contents, size_t size, size_t nmemb, void *userp)
{

    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

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

  int send(std::string data) {
    #if defined(__ANDROID__)
    __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature", "Msg=%s", data.c_str());
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
    char errbuf[CURL_ERROR_SIZE];
    long resp_code;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if(curl) {
      struct curl_slist *headers=NULL;
      headers=curl_slist_append(headers, "Content-Type: application/x-amz-json-1.1; charset=UTF-8");
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
      curl_easy_setopt(curl, CURLOPT_URL, CALL_HOME_URL.c_str());
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
      curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
      curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);

      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

      char *s = curl_easy_escape(curl, data.c_str(), data.length());       
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, s);

      errbuf[0] = 0;
      res = curl_easy_perform(curl);
      __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature ", "Resp str =%s",readBuffer.c_str() );
      __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature ", "Rest resp =%d", res);
      __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature ", "Rest resp err msg =%s", errbuf);

      if(res != CURLE_OK) {
        LOG(INFO) << "Rest client perform return code :" << res; 
        __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature ", "Rest resp NotOK");
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp_code);
        LOG(INFO) << "Rest client response code :" << resp_code;
        __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature ", "Rest resp =%ld", resp_code);
      } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp_code);
        LOG(INFO) << "Rest client response code :" << resp_code;
        __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature ", "Rest resp =%ld", resp_code);
      } 
      curl_free(s);
      curl_easy_cleanup(curl);
      curl_slist_free_all(headers);
    }
    curl_global_cleanup();
    return resp_code;
    #endif
    return 0;
  };
 private:
};

#endif //RESTCLIENT_H
