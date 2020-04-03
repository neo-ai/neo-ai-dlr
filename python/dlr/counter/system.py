"""System interface"""
import platform
import uuid
import logging
import abc

from .deviceinfo import DeviceInfo
from .utils.helper import get_hash_string


# Interface
class System:
    """Root Interface of Systems hierarchy"""
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
class ARM(System):
    """ARM systems interface"""
    @abc.abstractmethod
    def get_device_info(self):
        """Return a list of device information"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        raise NotImplementedError


class Linux_ARM(ARM):
    """Linux ARM system class"""
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
        except Exception as ex:
            logging.exception("linux_arm api exception occurred", exc_info=False)
            raise ex

    def get_device_info(self):
        """Return a list of fields of device information"""
        return self._device.get_info()

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        return self._device.uuid


class Android(ARM):
    """Android system interface"""
    @abc.abstractmethod
    def get_device_info(self):
        """Return a list of device information"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        raise NotImplementedError


class X86(System):
    """X86 architecture system interface"""
    @abc.abstractmethod
    def get_device_info(self):
        """Return a list of device information"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        raise NotImplementedError


class Linux_x86(X86):
    """Linux system on X86 arch class"""
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
        except Exception as ex:
            logging.exception("linux_x86 api exception occurred", exc_info=False)
            raise ex

    def get_device_info(self):
        """Return a list of fields of device information"""
        return self._device.get_info()

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        return self._device.uuid


class X64(System):
    """X64 architecture system"""
    @abc.abstractmethod
    def get_device_info(self):
        """Return a list of device information"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        raise NotImplementedError


class Linux(X64):
    """Linux system based on x64 architecture"""
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
        except Exception as ex:
            logging.exception("linux 64 api exception occurred", exc_info=False)
            raise ex

    def get_device_info(self):
        """Return a list of fields of device information"""
        return self._device.get_info()

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        return self._device.uuid


# mapped system types
SYSTEM_LIST = ["Linux_ARM", "Linux_x86", "Linux"]


# factory class for System wrapper class
class Factory:
    """Factory pattern return supported system instance"""
    @staticmethod
    def get_system(sys_typ):
        """Return instance of System as per operating system type"""
        try:
            map_sys_typ = [item for item in SYSTEM_LIST if item.lower() in sys_typ.lower()]
            if map_sys_typ:
                system_class = globals()[map_sys_typ[0]]
                return system_class()
            else:
                os_name = platform.system()
                map_sys_typ = [item for item in SYSTEM_LIST if item.lower() in os_name.lower()]
                if map_sys_typ:
                    system_class = globals()[map_sys_typ[0]]
                    return system_class()
        except Exception:
            logging.exception("unable to create system class instance", exc_info=False)
