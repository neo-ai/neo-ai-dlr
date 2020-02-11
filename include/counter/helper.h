#ifndef HELPER_H
#define HELPER_H

inline std::string get_hash_string(std::string& str)
{
  std::size_t sz_hash = std::hash<std::string>{}(str);
  std::string hash_str(std::to_string(sz_hash).c_str()); 
  return hash_str;
}
#endif // HELPER_H
