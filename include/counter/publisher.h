#ifndef PUBLISHER_H
#define PUBLISHER_H

#include <thread>
#include <queue>
#include <dmlc/logging.h>
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
  void send(const std::string str) {
    msg_que.push(str);
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
    thrd = nullptr;
    retrycnt = 0;
  }
  ~MsgPublisher();
  MsgPublisher(const MsgPublisher&){}
  MsgPublisher& operator=(const MsgPublisher& obj) {return *this;}
  RestClient *restcon;
  std::thread *thrd;
  std::queue<std::string> msg_que;
  bool stop_process;
  int retrycnt;
};

#endif //PUBLISHER_H
