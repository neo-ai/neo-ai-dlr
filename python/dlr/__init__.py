import _dlr
import sys

class Logger(object):
    def __init__(self, filename="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


class DLRModel(_dlr.DLRModel):

    def __init__(self, model_path, dev_type="cpu", dev_id=0, error_log_file="error.log"):
        logger = Logger(filename=error_log_file)
        sys.stdout = logger
        sys.stderr = logger
        super().__init__(model_path, dev_type, dev_id)