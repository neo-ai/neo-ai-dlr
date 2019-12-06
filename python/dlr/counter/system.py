import platform
import uuid
import hashlib

from .deviceinfo import DeviceInfo
from .utils.dlrlogger import logger


# wrapper class as per operating system
class System(object):
    def __init__(self):
        """create a instance of DeviceInfo() type"""
        self._device = DeviceInfo()

    def get_device_info(self):
        """Return a list of device information"""
        return self._device.get_info()

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        return self._device.uuid


class ARM(System):
    def __init__(self):
        System.__init__(self)

    def get_device_info(self):
        """Return a list of fields of device information"""
        pass

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        pass


class Linux(ARM):
    def __init__(self):
        ARM.__init__(self)
        try:
            # retrieve device information
            self._device.machine = platform.machine()
            self._device.arch = platform.architecture()[0]
            _uuid = ':'.join(
                ['{:02x}'.format((uuid.getnode() >> ele) & 0xff) for ele in range(0, 8 * 6, 8)][::-1])
            _md5uuid = hashlib.md5(_uuid.encode())
            self._device.uuid = str(_md5uuid.hexdigest())
            self._device.osname = platform.system()
            dist = platform.dist()
            self._device.dist = " ".join(x for x in dist)
            self._device.name = platform.node()
        except Exception as e:
            logger.exception("linux api exception occurred", exc_info=True)

    def get_device_info(self):
        """Return a list of fields of device information"""
        return System.get_device_info(self)

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        return self._device.uuid


class Raspbian(ARM):
    def __init__(self):
        ARM.__init__(self)

    def get_device_info(self):
        """Return a list of fields of device information"""
        pass

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        pass



class Android(ARM):
    def __init__(self):
        ARM.__init__(self)

    def get_device_info(self):
        """Return a list of fields of device information"""
        pass

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        pass


class X86(System):
    def __init__(self):
        System.__init__(self)

    def get_device_info(self):
        """Return a list of fields of device information"""
        pass

    def get_device_uuid(self):
        """Return DeviceInfo uuid"""
        pass


# factory class for System wrapper class
class Factory:
    @staticmethod
    def get_system(sys_typ):
        """Return instance of System as per operating system type"""
        try:
            system_class = globals()[sys_typ]
            return system_class()
        except Exception as e:
            logger.exception("unable to create system class instance")

