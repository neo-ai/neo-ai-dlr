##############
Installing DLR
##############

.. contents:: Contents
  :local:
  :backlinks: none

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

Building on Linux
=================

Ensure that all necessary software packages are installed: GCC (or Clang), CMake, and Python. For example, in Ubuntu, you can run

.. code-block:: bash

  sudo apt-get update
  sudo apt-get install -y python3 python3-pip gcc build-essential cmake

To build, create a subdirectory ``build`` and invoke CMake:

.. code-block:: bash

  mkdir build
  cd build
  cmake ..

Once CMake is done generating a Makefile, run GNU Make to compile:

.. code-block:: bash

  make -j4         # Use 4 cores to compile sources in parallel

By default, DLR will be built with CPU support only. To enable support for NVIDIA GPUs, enable CUDA, CUDNN, and TensorRT by calling CMake with extra options:

.. code-block:: bash

  cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_TENSORRT=/path/to/TensorRT/ 
  make -j4

You will need to install NVIDIA CUDA and TensorRT toolkits and drivers beforehand.

Similarly, to enable support for OpenCL devices, run CMake with:

.. code-block:: bash

  cmake .. -DUSE_OPENCL=ON 
  make -j4

Once the compilation is completed, install the Python package by running ``setup.py``:

.. code-block:: bash

  cd python
  python3 setup.py install --user

Building on Mac OS X
====================

Install GCC and CMake from `Homebrew <https://brew.sh/>`_:

.. code-block:: bash

  brew update
  brew install cmake gcc@8

To ensure that Homebrew GCC is used (instead of default Apple compiler), specify environment variables ``CC`` and ``CXX`` when invoking CMake:

.. code-block:: bash

  mkbir build
  cd build
  CC=gcc-8 CXX=g++-8 cmake ..
  make -j4

NVIDIA GPUs are not supported for Mac OS X target.

Once the compilation is completed, install the Python package by running ``setup.py``:

.. code-block:: bash

  cd python
  python3 setup.py install --user --prefix=''

Building on Windows
===================

DLR requires `Visual Studio 2017 <https://visualstudio.microsoft.com/downloads/>`_ as well as `CMake <https://cmake.org/>`_.

In the DLR directory, first run CMake to generate a Visual Studio project:

.. code-block:: cmd

  mkdir build
  cd build
  cmake .. -G"Visual Studio 15 2017 Win64"

If CMake run was successful, you should be able to find the solution file ``dlr.sln``. Open it with Visual Studio. To build, choose **Build Solution** on the **Build** menu.

NVIDIA GPUs are not yet supported for Windows target.

Once the compilation is completed, install the Python package by running ``setup.py``:

.. code-block:: cmd

  cd python
  python3 setup.py install --user

Building for Android on ARM
====================

Android build requires `Android NDK <https://developer.android.com/ndk/downloads/>`_. We utilize the android.toolchain.cmake file in NDK package to configure the crosscompiler 

Also required is `NDK standlone toolchain <https://developer.android.com/ndk/guides/standalone_toolchain>`_. Follow the instructions to generate necessary build-essential tools.

Once done with above steps, invoke cmake with following commands to build Android shared lib:

.. code-block:: bash

  cmake .. -DANDROID_BUILD=ON -DNDK_ROOT=/path/to/your/ndk/folder -DCMAKE_TOOLCHAIN_FILE=/path/to/your/ndk/folder/build/cmake/android.toolchain.cmake 
  make -j4

************************
Validation
************************

Validation on Linux
===================
.. code-block:: cmd

  cd tests/python/integration/
  python load_and_run_tvm_model.py
  python load_and_run_treelite_model.py