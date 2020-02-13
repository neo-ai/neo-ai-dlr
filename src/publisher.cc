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
  __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature", " ## Publisher Thread started ##");
  while (!stop_process) {
    if (!msg_que.empty()) {
      __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature", " queue is not empty ...");
      std::string msg = msg_que.front();
      if (retrycnt < CALL_HOME_REQ_STOP_MAX_COUNT) {
        // __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature", " # before sending message = %s", msg.c_str());
        int status = restcon->send(msg);
        if (status != 200) retrycnt++;
      }
      msg_que.pop();
    }
  }
    __android_log_print(ANDROID_LOG_DEBUG, "DLR Call Home Feature", " ## publisher thread exiting ! ##");
}

MsgPublisher* MsgPublisher::msgpublisher = nullptr;
