#include <mutex>
#include "counter/publisher.h"
std::mutex g_pub_mutex;

MsgPublisher* MsgPublisher::get_instance()
{
  g_pub_mutex.lock();
  if (!msgpublisher)
  {
    msgpublisher = new MsgPublisher();
    if (msgpublisher) {
      msgpublisher->thrd = new std::thread(&MsgPublisher::process_queue, msgpublisher);
    }
  }
  g_pub_mutex.unlock();
  return msgpublisher;
};

MsgPublisher::~MsgPublisher()
{
  delete thrd;
  delete restcon;
};

void MsgPublisher::process_queue()
{
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
