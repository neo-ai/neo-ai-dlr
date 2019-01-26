import os
from setuptools import setup, find_packages
from subprocess import check_output
from setuptools.dist import Distribution

data_files = []
for path, dirnames, filenames in os.walk('python'):
    for filename in filenames:
        data_files.append(os.path.join(path, filename))
data_files.append('../build/lib/libdlr.so')


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


setup(
    name="dlr",
    version="1.0",

    zip_safe=False,
    install_requires=['numpy', 'decorator'],

    distclass=BinaryDistribution,

    # declare your packages
    packages=find_packages(),

    # include data files
    include_package_data=True,
    data_files=[('dlr', data_files)],

    root_script_source_version='python3.4',
    default_python='python3.4',
)
