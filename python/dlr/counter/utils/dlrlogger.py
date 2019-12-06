import logging


logging.basicConfig(filename='ccm_app.log', filemode='w', format='%(asctime)s-%(levelname)s-%(message)s')
# Creating an object
logger=logging.getLogger()
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
