#ifndef PUBLISHER_H
#define PUBLISHER_H

#include <thread>
#include <queue>
#include <fstream>
#include <ostream>
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
  ~MsgPublisher();
  void send(std::string& str) {
    msg_que.push(str);
  }
  void process_queue();
 private:
  MsgPublisher() {
    restcon = new RestClient();
    if (!restcon) {stop_process = true; LOG(FATAL) << "Message Publisher object null !";}
    stop_process = false;
    retrycnt = 0;
  };
  RestClient *restcon;
  std::thread *thrd;
  std::queue<std::string> msg_que;
  bool stop_process;
  int retrycnt;
};

#endif //PUBLISHER_H
