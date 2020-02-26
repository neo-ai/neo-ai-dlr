"""Helper Module : Holding common utility functions"""
import hashlib


def get_hash_string(enc_model):
    """
    Encode model name using md5 hashing.
    Parameters
    ----------
    enc_model: str

    Returns
    -------
    str:
        return encoded model name
    """
    md5str = hashlib.md5(enc_model)
    return md5str
