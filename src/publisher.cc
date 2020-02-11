#include <mutex>
#include "counter/publisher.h"

MsgPublisher* MsgPublisher::get_instance() {
  if (!msgpublisher) {
    msgpublisher = new MsgPublisher();
    if (msgpublisher) {
      msgpublisher->thrd = new std::thread(&MsgPublisher::process_queue, msgpublisher);
    }
  }
  return msgpublisher;
};

MsgPublisher::~MsgPublisher() {
  delete thrd;
  thrd = nullptr;
  delete restcon;
  restcon = nullptr;
};

void MsgPublisher::process_queue() {
  while (!stop_process) {
    while(!msg_que.empty()) {
      std::string msg = msg_que.front();
      if (retrycnt < CALL_HOME_REQ_STOP_MAX_COUNT) {
        int status = restcon->send(msg);
        if (status != 200) retrycnt++;
      }
      msg_que.pop();
    }
  }
}

MsgPublisher* MsgPublisher::msgpublisher = nullptr;
