import os
import io
import sys
from setuptools import setup, find_packages
from subprocess import check_output
from setuptools.dist import Distribution
from platform import system

if "--universal" in sys.argv:
  raise ValueError("Creating py2.py3 wheels is not supported")

CURRENT_DIR = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

BUILD_DIR = "../build/lib/"

libname = 'libdlr.so'
if sys.platform == 'win32':
  libname = 'dlr.dll'
elif sys.platform == 'darwin':
  libname = 'libdlr.dylib'

LIB_PATH = os.path.join(BUILD_DIR, libname)

if os.path.exists(LIB_PATH):
  print("Found", libname, "at", LIB_PATH)
  include_package_data = True
  data_files = [('dlr', [LIB_PATH,])]
else:
  print(libname, "is not found!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("!!! Preparing universal py3 version of DLR !!!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  include_package_data = False
  data_files = None

# fetch meta data
METADATA_PY = os.path.abspath("./dlr/metadata.py")
METADATA_PATH = {"__file__": METADATA_PY}
METADATA_BIN = open(METADATA_PY, "rb")
exec(compile(METADATA_BIN.read(), METADATA_PY, 'exec'), METADATA_PATH, METADATA_PATH)
METADATA_BIN.close()

setup(
    name="dlr",
    version=METADATA_PATH['VERSION'],

    zip_safe=False,
    install_requires=['numpy', 'requests', "distro"],

    # declare your packages
    packages=find_packages(),

    # include data files
    include_package_data=include_package_data,
    data_files=data_files,

    description = 'Common runtime for machine learning models compiled by \
        AWS SageMaker Neo, TVM, or TreeLite.',
    long_description=io.open(os.path.join(CURRENT_DIR, '../README.md'), encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author = 'AWS Neo',
    author_email = 'aws-neo-ai@amazon.com',
    url='https://github.com/neo-ai/neo-ai-dlr',
    license = "Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires = '>=3.5',
)
