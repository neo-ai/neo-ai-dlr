#ifndef MY_APPLICATION_PUBLISHER_H
#define MY_APPLICATION_PUBLISHER_H

#include <thread>
#include <queue>
#include <fstream>
#include <ostream>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include "rest_client.h"

class MsgPublisher {
 public:
  static MsgPublisher* get_instance();
  ~MsgPublisher();
  void send(std::string str) {
    //__android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "inside msgpublisher send");
    msg_que.push(str);
  }
  void process_queue();
 private:
  MsgPublisher() {
   //__android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "inside msgpublisher ctor");
   restcon = new RestClient();
   stop_process = false;
  };
 private:
  RestClient *restcon;
  std::thread *thrd;
  std::queue<std::string> msg_que;
  bool stop_process;
 public:
  static MsgPublisher* msgpublisher;
};

#endif //MY_APPLICATION_PUBLISHER_H
