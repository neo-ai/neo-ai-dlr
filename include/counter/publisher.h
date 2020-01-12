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
  static MsgPublisher* msgpublisher;
  static MsgPublisher* get_instance();
  static void release_instance() {
    while (!msgpublisher->msg_que.empty());
    msgpublisher->stop_process = true;
    msgpublisher->thrd->join();
    delete msgpublisher; 
    msgpublisher = nullptr;
  }
  ~MsgPublisher();
  void send(std::string str) {
    msg_que.push(str);
  }
  void process_queue();
 private:
  MsgPublisher() {
    restcon = new RestClient();
    stop_process = false;
  };
  RestClient *restcon;
  std::thread *thrd;
  std::queue<std::string> msg_que;
  bool stop_process;
};

#endif //MY_APPLICATION_PUBLISHER_H
