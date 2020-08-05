import os
import io
from setuptools import setup, find_packages
from subprocess import check_output
from setuptools.dist import Distribution
from platform import system
from dlr import __version__

data_files = []
for path, dirnames, filenames in os.walk('python'):
    for filename in filenames:
        data_files.append(os.path.join(path, filename))

# Use libpath.py to locate libdlr.so
LIBPATH_PY = os.path.abspath('./dlr/libpath.py')
LIBPATH = {'__file__': LIBPATH_PY}
exec(compile(open(LIBPATH_PY, "rb").read(), LIBPATH_PY, 'exec'),
     LIBPATH, LIBPATH)

CURRENT_DIR = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
LIB_PATH = [os.path.relpath(LIBPATH['find_lib_path'](setup=True), CURRENT_DIR)]

if not LIB_PATH:
    raise RuntimeError('libdlr.so missing. Please compile first using CMake')

setup(
    name="dlr",
    version=__version__,

    zip_safe=False,
    install_requires=['numpy'],

    # declare your packages
    packages=find_packages(),

    # include data files
    include_package_data=True,
    data_files=[('dlr', LIB_PATH)],

    description = 'Common runtime for machine learning models compiled by \
        AWS SageMaker Neo, TVM, or TreeLite.',
    long_description=io.open(os.path.join(CURRENT_DIR, '../README.md'), encoding='utf-8').read(),
    author = 'AWS Neo',
    author_email = 'aws-neo-ai@amazon.com',
    url='https://github.com/neo-ai/neo-ai-dlr',
    license = "Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
)
