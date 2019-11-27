# a data structure for device specific information
class DeviceInfo:
    def __init__(self):
        self.machine = ""
        self.arch = ""
        self.osname = ""
        self.name = ""
        self.dist = ""
        self.uuid = ""
         

    # prepare a list of info
    def get_info(self):
        """
        Prepare a list of data member in sequence. 
        1. Machine 
        2. Architecture
        3. Operating system
        4. Machine name 
        5. Operating system distrubution
        6. UUID
        Parameters
        ----------
        ret : list
            return value from API calls
        """

        data_lst = []
        data_lst.append(self.machine)
        data_lst.append(self.arch)
        data_lst.append(self.osname)
        data_lst.append(self.name)
        data_lst.append(self.dist)
        data_lst.append(self.uuid)
        return data_lst


class ARMDevice(DeviceInfo):
    def __init__(self):
        DeviceInfo.__init__(self)
        self.processor = ""
        self.speed = ""
        self.arch = ""

    def get_info(self):
        """
        Prepare a list of data member in sequence. 
        1. Processor 
        2. Speed 
        3. Arch 
        Parameters
        ----------
        ret : list
            return value from API calls
        """
        arm_data_lst = []
        arm_data_lst.append(self.processor)
        arm_data_lst.append(self.speed)
        arm_data_lst.append(self.arch)
        return arm_data_lst
