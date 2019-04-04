import platform
import urllib

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

def get_arch():
    machine_type = platform.machine()
    if machine_type == 'x86_64':
        return 'x86_64'
    elif machine_type == 'aarch64':
        return 'ec2_a1'
    elif machine_type == 'armv7l':
        return 'rasp3b'
    else:
        raise ValueError('Unsupported platform, please supply matching model')

def get_models(model_name, arch):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            model_name)
    if not os.path.exists(model_path):
        try:  
            os.mkdir(model_path)
        except OSError:  
            raise ValueError("Creation of the directory %s failed" % path)
    
    s3_bucket = 'https://s3-us-west-2.amazonaws.com/neo-ai-dlr-test-artifacts'
    artifact_extensions = ['.json', '.params', '.so']
    print("Preparing model artifacts for %s ..." % model_name)
    for extension in artifact_extensions:
        s3_path = s3_bucket + '/' + model_name + '/' + arch + extension
        local_path = os.path.join(model_path, model_name + '_' + arch + extension)
        if not os.path.exists(local_path):
            try:
                urlretrieve(s3_path, local_path) 
            except urllib.error.URLError or urllib.error.HTTPError:
                raise ValueError('Downloading of model artifacts from %s failed' % s3_path)
                
