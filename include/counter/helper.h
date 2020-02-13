#ifndef HELPER_H
#define HELPER_H

#include <curl/md5.h>

inline std::string get_hash_string(const std::string& str)
{
  //std::size_t sz_hash = std::hash<std::string>{}(str);
  //std::string hash_str(std::to_string(sz_hash).c_str()); 


  unsigned char digest[16];
  //const char* string = "Hello World";
  //struct MD5Context context;
  struct MD5state_st context;
  MD5_Init(&context);
  //MD5Update(&context, string, strlen(string));
  MD5_Update(&context, str.c_str(), strlen(str.c_str()));
  MD5_Final(digest, &context);


  char md5string[33];
  for(int i = 0; i < 16; ++i)
    sprintf(&md5string[i*2], "%02x", (unsigned int)digest[i]);

   std::string hash_str = md5string;
  return hash_str;
}
#endif // HELPER_H
