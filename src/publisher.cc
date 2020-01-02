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
      restcon->send(msg);
      msg_que.pop();
    }
  }
}

MsgPublisher* MsgPublisher::msgpublisher = nullptr;
