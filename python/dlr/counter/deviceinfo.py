""" a data structure for device specific information"""


class DeviceInfo(object):
    """Holding Generic information about Device"""

    def __init__(self):
        self.machine = ""
        self.arch = ""
        self.osname = ""
        self.name = ""
        self.dist = ""
        self.uuid = ""

    def get_info(self):
        """
        Prepare a dictionary of data member in sequence.
        1. Machine
        2. Architecture
        3. Operating system
        4. Machine name
        5. Operating system distribution
        6. UUID
        Parameters
        ----------
        self

        Returns
        -------
        dictionary:
            return a dictionary of data members
        """

        dict_device = {
            "uuid": self.uuid,
            "machine": self.machine,
            "arch": self.arch,
            "os": self.osname,
            "os distribution": self.dist
        }
        return dict_device


class ARMDevice(DeviceInfo):
    """Holding ARM device information"""

    def __init__(self):
        DeviceInfo.__init__(self)
        self.processor = ""
        self.speed = ""
        self.arch = ""

    def get_info(self):
        """
        Prepare a dictionary of data member in sequence.
        1. Processor
        2. Speed
        3. Arch
        Parameters
        ----------
        self :
            return a dictionary of data members
        """
        dict_arm = {
            "processor": self.processor,
            "speed": self.speed,
            "arch": self.arch
        }

        return dict_arm
