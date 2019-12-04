import platform
import uuid
from .deviceinfo import DeviceInfo
from .utils.dlrlogger import logger
import hashlib

# wrapper class as per operating system
class System:
    def __init__(self):
        """create a instance of DeviceInfo() type"""
        self._device = DeviceInfo()

    def get_info(self):
        """Return a list of device information"""
        return self._device.get_info()

    def get_device(self):
        """Return DeviceInfo instance"""
        return self._device

    def retrieve_info(self):
        pass


class ARM(System):
    def __init__(self):
        System.__init__(self)

    def get_info(self):
        """Return a list of fields of device information"""
        self.retrieve_info()
        return System.get_info(self)

    def retrieve_info(self):
        """Retrieve device specific information from Linux/ARM"""
        try:
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
            logger.warning("System API exception occured!", exc_info=True)


class Raspbian(ARM):
    def __init__(self):
        ARM.__init__(self)

    def get_info(self):
        pass

    def retrieve_info(self):
        pass


class Android(ARM):
    def __init__(self):
        ARM.__init__(self)

    def get_info(self):
        pass

    def retrieve_info(self):
        pass


class X86(System):
    def __init__(self):
        System.__init__(self)

    def get_info(self):
        pass

    def retrieve_info(self):
        pass


# factory class for System wrapper class
class Factory:
    @staticmethod
    def get_system(sys_typ):
        """Return instance of System as per operating system type"""
        if sys_typ == 'Linux':
            return ARM()
        elif sys_typ == 'Android':
            return Android()
        else:
            # no system  wrapper available to retrieve info
            pass
