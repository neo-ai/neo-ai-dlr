##############
Installing DLR
##############

.. contents:: Contents
  :local:
  :backlinks: none

***********************************************
Installing Pre-built DLR Wheels for Your Device
***********************************************

DLR has been built and tested against devices in table 1. If you find your device(s) listed below, you can install DLR with the corresponding S3 link via 

  .. code-block:: bash

    pip install  link-to-matching-wheel

Table 1: List of Supported Devices
----------------------------------

+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Manufacturer | Device Name               |  Wheel URL                                                                                                                                                |
+==============+===========================+===========================================================================================================================================================+
| Acer         | TV AISage                 |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/acer-aarch64-linaro4_4_154-glibc2_24-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl            |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Amazon       | AWS a1                    |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/a1-aarch64-ubuntu18_04-glibc2_27-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl                |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Amazon       | AWS p2/p3/g4              |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/p3-x86_64-cu10-ubuntu18_04-glibc2_27-libstdpp3_4/dlr-1.2.0-py2.py3-none-any.whl             |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Amazon       | Deeplens                  |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.1.0/deeplens-x86_64-igp-ubuntu16_04-glibc2_23-libstdcpp3_4/dlr-1.1.0-py2.py3-none-any.whl       |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Rockchips    | RK3399                    |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/firefly-aarch64-mali-ubuntu16_04-glibc2_23-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl      |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| NVIDIA       | Jetson TX1 (JetPack 4.3)  |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/jetsontx1-aarch64-cu10-ubuntu18_04-glibc2_27-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl    |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| NVIDIA       | Jetson TX2 (JetPack 4.3)  |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/jetsontx2-aarch64-cu10-ubuntu18_04-glibc2_27-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl    |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| NVIDIA       | Jetson Nano (JetPack 4.3) |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/jetsonnano-aarch64-cu10-ubuntu18_04-glibc2_27-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl   |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| NVIDIA       |Jetson Xavier (JetPack 4.3)|  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/jetsonxavier-aarch64-cu10-ubuntu18_04-glibc2_27-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| Raspberry    | Rasp3b                    |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/pi-armv7l-raspbian4.14.71-glibc2_24-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl             |
+--------------+---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+

If your device is not listed in the table, use table2. You will identify your device by the processor architecture, operating system, and versions of GLIBC and LIBSTDC++. Of note, DLR installation may depend on other configuration differences or even location of dependency libraries; if the provided wheel URL does not work, please consider compiling DLR from source (see `Building DLR from source`_ section).

Table2: List of Supported Architectures (Incomplete)
----------------------------------------------------

+------------------------+--------------+---------------+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------+ 
| Processor architecture | OS           | GLIBC version | LIBSTDC++ version | Wheel URL                                                                                                                                  | 
+========================+==============+===============+===================+============================================================================================================================================+ 
| aarch64                | Ubuntu 18.04 | 2.27+         | 3.4+              |  https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/a1-aarch64-ubuntu18_04-glibc2_27-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl | 
+------------------------+--------------+---------------+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------+ 
| armv7l                 | Debian 9.0   | 2.24+         | 3.4+              |https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.2.0/pi-armv7l-raspbian4.14.71-glibc2_24-libstdcpp3_4/dlr-1.2.0-py2.py3-none-any.whl| 
+------------------------+--------------+---------------+-------------------+--------------------------------------------------------------------------------------------------------------------------------------------+ 


************************
Building DLR from source
************************

Building DLR consists of two steps:

1. Build the shared library from C++ code (``libdlr.so`` for Linux, ``libdlr.dylib`` for Mac OSX, and ``dlr.dll`` for Windows).
2. Then install the Python package ``dlr``.

.. note:: Use of Git submodules

  DLR uses Git submodules to manage dependencies. So when you clone the repo, remember to specify ``--recursive`` option:
  
  .. code-block:: bash

    git clone --recursive https://github.com/neo-ai/neo-ai-dlr
    cd neo-ai-dlr

Building on Linux
-----------------

Requirements
""""""""""""

Ensure that all necessary software packages are installed: GCC (or Clang), CMake, and Python. For example, in Ubuntu, you can run

.. code-block:: bash

  sudo apt-get update
  sudo apt-get install -y python3 python3-distutils build-essential cmake curl ca-certificates
  curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  sudo python3 /tmp/get-pip.py
  rm /tmp/get-pip.py
  sudo pip3 install -U pip setuptools wheel


Building for CPU
""""""""""""""""

First, clone the repository.

.. code-block:: bash

  git clone --recursive https://github.com/neo-ai/neo-ai-dlr
  cd neo-ai-dlr

Create a subdirectory ``build``:

.. code-block:: bash

  mkdir build
  cd build

Invoke CMake to generate a Makefile and then run GNU Make to compile:

.. code-block:: bash

  cmake ..
  make -j4         # Use 4 cores to compile sources in parallel

Once the compilation is completed, install the Python package by running ``setup.py``:

.. code-block:: bash

  cd ../python
  python3 setup.py install --user

Building for NVIDIA GPU on Jetson Devices
"""""""""""""""""""""""""""""""""""""""""

By default, DLR will be built with CPU support only. To enable support for NVIDIA GPUs, enable CUDA, CUDNN, and TensorRT by calling CMake with these extra options.

DLR requires CMake 3.13 or greater. First, we will build CMake from source.

.. code-block:: bash

  sudo apt-get install libssl-dev
  wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2.tar.gz
  tar xvf cmake-3.17.2.tar.gz
  cd cmake-3.17.2
  ./bootstrap
  make -j4
  sudo make install

Now, build DLR.

.. code-block:: bash
 
  git clone --recursive https://github.com/neo-ai/neo-ai-dlr
  cd neo-ai-dlr
  mkdir build
  cd build
  cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_TENSORRT=ON
  make -j4
  cd ../python
  python3 setup.py install --user

Building for NVIDIA GPU (Cloud or Desktop)
""""""""""""""""""""""""""""""""""""""""""

By default, DLR will be built with CPU support only. To enable support for NVIDIA GPUs, enable CUDA, CUDNN, and TensorRT by calling CMake with these extra options.

If you do not have a system install of TensorRT, first download the relevant .tar.gz file from https://developer.nvidia.com/nvidia-tensorrt-download
Please follow instructions from https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar to install TensorRT.
Now, provide the extracted .tar.gz folder path to ``-DUSE_TENSORRT`` when configuring cmake.

If you have a system install of TensorRT via Deb or RPM package, you can instead use ``-DUSE_TENSORRT=ON`` which will find the install directory automatically.

.. code-block:: bash

  git clone --recursive https://github.com/neo-ai/neo-ai-dlr
  cd neo-ai-dlr
  mkdir build
  cd build
  cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_TENSORRT=/path/to/TensorRT/ 
  make -j4
  cd ../python
  python3 setup.py install --user

See `Additional Options for TensorRT Optimized Models <https://neo-ai-dlr.readthedocs.io/en/latest/tensorrt.html>`_ to learn how to enable FP16 precision and more for your Neo optimized models which use TensorRT.


Building for OpenCL Devices
"""""""""""""""""""""""""""

Similarly, to enable support for OpenCL devices, run CMake with ``-DUSE_OPENCL=ON``:

.. code-block:: bash

  git clone --recursive https://github.com/neo-ai/neo-ai-dlr
  cd neo-ai-dlr
  mkdir build
  cd build
  cmake .. -DUSE_OPENCL=ON 
  make -j4
  cd ../python
  python3 setup.py install --user

Building on Mac OS X
--------------------

Install GCC and CMake from `Homebrew <https://brew.sh/>`_:

.. code-block:: bash

  brew update
  brew install cmake gcc@8

To ensure that Homebrew GCC is used (instead of default Apple compiler), specify environment variables ``CC`` and ``CXX`` when invoking CMake:

.. code-block:: bash

  git clone --recursive https://github.com/neo-ai/neo-ai-dlr
  cd neo-ai-dlr
  mkdir build
  cd build
  CC=gcc-8 CXX=g++-8 cmake ..
  make -j4

NVIDIA GPUs are not supported for Mac OS X target.

Once the compilation is completed, install the Python package by running ``setup.py``:

.. code-block:: bash

  cd ../python
  python3 setup.py install --user --prefix=''

Building on Windows
-------------------

DLR requires `Visual Studio 2017 <https://visualstudio.microsoft.com/downloads/>`_ as well as `CMake <https://cmake.org/>`_.

In the DLR directory, first run CMake to generate a Visual Studio project:

.. code-block:: bash

  git clone --recursive https://github.com/neo-ai/neo-ai-dlr
  cd neo-ai-dlr
  mkdir build
  cd build
  cmake .. -G"Visual Studio 15 2017 Win64"

If CMake run was successful, you should be able to find the solution file ``dlr.sln``. Open it with Visual Studio. To build, choose **Build Solution** on the **Build** menu.

NVIDIA GPUs are not yet supported for Windows target.

Once the compilation is completed, install the Python package by running ``setup.py``:

.. code-block:: bash

  cd ../python
  python3 setup.py install --user

Building for Android on ARM
---------------------------

Android build requires `Android NDK <https://developer.android.com/ndk/downloads/>`_. We utilize the android.toolchain.cmake file in NDK package to configure the crosscompiler 

Also required is `NDK standlone toolchain <https://developer.android.com/ndk/guides/standalone_toolchain>`_. Follow the instructions to generate necessary build-essential tools.

Once done with above steps, invoke cmake with following commands to build Android shared lib:

.. code-block:: bash

  git clone --recursive https://github.com/neo-ai/neo-ai-dlr
  cd neo-ai-dlr
  mkdir build
  cd build
  cmake .. -DANDROID_BUILD=ON \
    -DNDK_ROOT=/path/to/your/ndk/folder \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/your/ndk/folder/build/cmake/android.toolchain.cmake \
    -DANDROID_PLATFORM=android-21

  make -j4

``ANDROID_PLATFORM`` should correspond to ``minSdkVersion`` of your project. If ``ANDROID_PLATFORM`` is not set it will default to ``android-21``.

For arm64 targets, add 

.. code-block:: bash

  -DANDROID_ABI=arm64-v8a 
  
to cmake flags.

Building DLR with TFLite
------------------------
DLR build can include ``libtensorflow-lite.a`` library into ``libdlr.so`` shared library.

Currently DLR supports TFLite 1.15.2 (branch r1.15).
Build ``libtensorflow-lite.a`` as explained `here <https://www.tensorflow.org/lite/guide/build_arm64>`_

To build ``libtensorflow-lite.a`` for Android you can look at this `docs <https://gist.github.com/apivovarov/9f67fc02b84cf6d139c05aa1a8bc16f9>`_

To build DLR with TFLite use cmake flag ``WITH_TENSORFLOW_LITE_LIB``, e.g.

.. code-block:: bash

  cmake .. \
  -DWITH_TENSORFLOW_LITE_LIB=/opt/tensorflow-1.15/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a

To test DLR with TFLite use ``dlr_tflite_test``

.. code-block:: bash

  ./dlr_tflite_test


Building for Android Archive (AAR) file
---------------------------------------

Install `Android Studio <https://developer.android.com/studio>`_.

.. code-block:: bash

  cd aar
  # create file local.properties
  # put line containing path to Android/sdk
  # sdk.dir=/Users/root/Library/Android/sdk

  # Run gradle build
  ./gradlew assembleRelease

  # dlr-release.aar file will be under dlr/build/outputs/aar/ folder
  ls -lah dlr/build/outputs/aar/dlr-release.aar


Building DLR with Hexagon support
---------------------------------

To build DLR with Hexagon compiled models support use flag ``-DWITH_HEXAGON=1``

.. code-block:: bash

  cmake .. -DWITH_HEXAGON=1

.. code-block:: bash

  ./dlr_hexagon_test


***********************************
Validation After Build (Linux Only)
***********************************

.. code-block:: bash

  cd tests/python/integration/
  python load_and_run_tvm_model.py
  python load_and_run_treelite_model.py
