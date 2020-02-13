#ifndef PUBLISHER_H
#define PUBLISHER_H

#include <thread>
#include <queue>
#include <dmlc/logging.h>
#include <android/log.h>
#include "rest_client.h"

/*! \brief class MsgPublisher
 */
class MsgPublisher {
 public:
  static MsgPublisher* msgpublisher;
  static MsgPublisher* get_instance();
  static void release_instance() {
    msgpublisher->stop_process = true;
    msgpublisher->thrd->join();
    delete msgpublisher; 
    msgpublisher = nullptr;
  }
  ~MsgPublisher();
  void send(const std::string str) {
    //__android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature", "# pushing to queue =%s#", str.c_str());
    msg_que.push(str);
    //__android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature", "# queue size =%u", msg_que.size());
  }
  void process_queue();
 private:
  MsgPublisher() {
    restcon = new RestClient();
    if (!restcon) { 
      stop_process = true; 
      LOG(FATAL) << "Message Publisher object null !";
      throw std::runtime_error("Message Publisher object null !");
    }
    stop_process = false;
    retrycnt = 0;
  }
  MsgPublisher(const MsgPublisher&){}
  MsgPublisher& operator=(const MsgPublisher& obj) {return *this;}
  RestClient *restcon;
  std::thread *thrd;
  std::queue<std::string> msg_que;
  bool stop_process;
  int retrycnt;
};

#endif //PUBLISHER_H
