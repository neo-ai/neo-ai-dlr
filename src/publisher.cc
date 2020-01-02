#include "counter/publisher.h"

MsgPublisher* MsgPublisher::get_instance()
{
  if (!msgpublisher)
  {
    msgpublisher = new MsgPublisher();
    if (msgpublisher) {
      msgpublisher->thrd = new std::thread(&MsgPublisher::process_queue, msgpublisher);
    }
  }
  return msgpublisher;
};

MsgPublisher::~MsgPublisher()
{
  //__android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "inside msgpublisher dtor");
  while (!msg_que.empty());
  stop_process = true;
  thrd->join();
  delete thrd;
  delete restcon;
};

void MsgPublisher::process_queue()
{
  while (!stop_process) {
    while(!msg_que.empty()) {
      std::string msg = msg_que.front();
      //__android_log_print(ANDROID_LOG_DEBUG, "MyAPP", "inside msgpublisher while 2, %s", msg.c_str());
      restcon->send(msg);
      msg_que.pop();
    }
  }
}

MsgPublisher* MsgPublisher::msgpublisher = nullptr;
