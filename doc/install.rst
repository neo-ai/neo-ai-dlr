##############
Installing DLR
##############

.. contents:: Contents
  :local:
  :backlinks: none

***********************************************
Installing Pre-built DLR Wheels for Your Device
***********************************************

DLR has been built and tested against many devices. You can install DLR with the corresponding S3 link via

  .. code-block:: bash

    pip install  link-to-matching-wheel

Please see [Releases](https://github.com/neo-ai/neo-ai-dlr/releases) to download DLR wheels for each DLR release.

************************
Building DLR from source
************************

Building DLR consists of two steps:

1. Build the shared library from C++ code (``libdlr.so`` for Linux, ``libdlr.dylib`` for macOS, and ``dlr.dll`` for Windows).
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

DLR requires CMake 3.17.2 or greater. First, we will build CMake from source.

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

Building on macOS
--------------------

Install CMake from `Homebrew <https://brew.sh/>`_:

.. code-block:: bash

  brew update
  brew install cmake

.. code-block:: bash

  git clone --recursive https://github.com/neo-ai/neo-ai-dlr
  cd neo-ai-dlr
  mkdir build
  cd build
  cmake ..
  make -j4

NVIDIA GPUs are not supported for macOS target.

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


Building DLR with TensorFlow C library
--------------------------------------

We can use DLR to run Tensorflow 1.x / 2.x models (including `TensorRT converted models <https://blog.tensorflow.org/2021/01/leveraging-tensorflow-tensorrt-integration.html>`_) via TensorFlow C library.

TensorFlow C library can be downloaded from `tensorflow.org <https://www.tensorflow.org/install/lang_c>`_ or built `from source <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README.md>`_.

To build DLR with TensorFlow C library:

.. code-block:: bash

  # Build DLR with libtensorflow
  cmake .. -DWITH_TENSORFLOW_LIB=<path to libtensorflow folder>
  make -j8

  # Test DLR with libtensorflow
  export LD_LIBRARY_PATH=<path to libtensorflow folder>/lib
  ./dlr_tensorflow_test


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
