#ifndef HELPER_H
#define HELPER_H

#include <curl/md5.h>

inline std::string get_hash_string(const std::string& str)
{
  #if defined(__ANDROID__)
  unsigned char digest[16];
  struct MD5state_st context;
  MD5_Init(&context);
  MD5_Update(&context, str.c_str(), strlen(str.c_str()));
  MD5_Final(digest, &context);
  char md5string[33];
  for(int i = 0; i < 16; ++i)
    sprintf(&md5string[i*2], "%02x", (unsigned int)digest[i]);
  std::string hash_str = md5string;
  return hash_str;
  #endif
  return "";
}
#endif // HELPER_H
