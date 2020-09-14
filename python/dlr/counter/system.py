"""System interface"""
import platform
import uuid
import logging
import abc
import distro


from .deviceinfo import DeviceInfo
from .utils.helper import get_hash_string


class Linux:
    """Linux system based on x64 architecture"""

    def __init__(self):
        self._device = DeviceInfo()

        try:
            # retrieve device information
            self._device.machine = platform.machine()
            self._device.arch = platform.architecture()[0]
            _uuid = ":".join(
                [
                    "{:02x}".format((uuid.getnode() >> ele) & 0xFF)
                    for ele in range(0, 8 * 6, 8)
                ][::-1]
            )
            _md5uuid = get_hash_string(_uuid.encode())
            self._device.uuid = str(_md5uuid.hexdigest())
            self._device.osname = platform.system()
            dist = distro.linux_distribution()
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
# not-support android and non-linux base system
SUPPORTED_SYSTEM_LIST = ["LINUX_ARM", "LINUX_X86", "LINUX"]


# factory class for System wrapper class
class Factory:
    """Factory pattern return supported system instance"""

    @staticmethod
    def get_system(sys_typ):
        """Return instance of System as per operating system type"""
        try:
            os_name = platform.system()
            map_sys_typ = [
                item
                for item in SUPPORTED_SYSTEM_LIST
                if item.lower() in os_name.lower()
            ]
            system_class = Linux() if map_sys_typ else None
            return system_class

        except Exception:
            logging.exception("unable to create system class instance", exc_info=False)
