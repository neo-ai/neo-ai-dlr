import platform
import uuid
import logging
import abc
from abc import ABC

from .deviceinfo import DeviceInfo
from .utils.helper import *

# Interface
class System:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_device_info(self):
        """Return a list of device information"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        raise NotImplementedError


# Wrapper class
class ARM(System, ABC):
    pass


class Linux_ARM(ARM):
    def __init__(self):
        self._device = DeviceInfo()

        try:
            # retrieve device information
            self._device.machine = platform.machine()
            self._device.arch = platform.architecture()[0]
            _uuid = ':'.join(
                ['{:02x}'.format((uuid.getnode() >> ele) & 0xff) for ele in range(0, 8 * 6, 8)][::-1])
            _md5uuid = get_hash_string(_uuid.encode()) 
            self._device.uuid = str(_md5uuid.hexdigest())
            self._device.osname = platform.system()
            dist = platform.dist()
            self._device.dist = " ".join(x for x in dist)
            self._device.name = platform.node()
        except Exception as e:
            logging.exception("linux_arm api exception occurred", exc_info=False)
            raise e

    def get_device_info(self):
        """Return a list of fields of device information"""
        return self._device.get_info()

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        return self._device.uuid


class Android(ARM, ABC):
    pass


class X86(System, ABC):
    pass


class Linux_x86(X86):
    def __init__(self):
        self._device = DeviceInfo()

        try:
            # retrieve device information
            self._device.machine = platform.machine()
            self._device.arch = platform.architecture()[0]
            _uuid = ':'.join(
                ['{:02x}'.format((uuid.getnode() >> ele) & 0xff) for ele in range(0, 8 * 6, 8)][::-1])
            _md5uuid = get_hash_string(_uuid.encode()) 
            self._device.uuid = str(_md5uuid.hexdigest())
            self._device.osname = platform.system()
            dist = platform.dist()
            self._device.dist = " ".join(x for x in dist)
            self._device.name = platform.node()
        except Exception as e:
            logging.exception("linux_x86 api exception occurred", exc_info=False)
            raise e

    def get_device_info(self):
        """Return a list of fields of device information"""
        return self._device.get_info()

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        return self._device.uuid


# mapped system types
system_list = ["Linux_ARM", "Linux_x86"]


# factory class for System wrapper class
class Factory:
    @staticmethod
    def get_system(sys_typ):
        """Return instance of System as per operating system type"""
        try:
            map_sys_typ = [item for item in system_list if item.lower() in sys_typ.lower()]
            if map_sys_typ:
                system_class = globals()[map_sys_typ[0]]
                return system_class()
        except Exception as e:
            logging.exception("unable to create system class instance")
