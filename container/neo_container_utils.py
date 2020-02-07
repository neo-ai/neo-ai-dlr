SHAPES_FILE = 'model-shapes.json'
SUPPORTED_CONTENT_TYPE = ['image/jpeg', 'image/png', 'application/x-image']
'''
Trusted error logs are mounted by default on the algorithm container's volume. 
Any logs written to /opt/ml/errors/errors.log will be reported back to sagemaker.
DO NOT CHANGE THE LOG FILE PATH/NAME
'''
SAGEMAKER_ERROR_LOG_FILE = "/opt/ml/errors/errors.log"

