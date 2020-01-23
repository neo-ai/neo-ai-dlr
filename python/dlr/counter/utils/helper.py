import hashlib

def get_hash_string(encstr):
  md5str = hashlib.md5(encstr)
  return md5str   
