################################################
Additional Options for TensorRT Optimized Models
################################################

.. contents:: Contents
  :local:
  :backlinks: none

***************
TensorRT in Neo
***************

For targets with NVIDIA GPUs, Neo may use `TensorRT <https://developer.nvidia.com/tensorrt>`_ to optimize all or part of your model.
If your optimized model is using TensorRT, you will see outputs similar to the following
during the first inference after loading the model.


.. code-block:: none

  Building new TensorRT engine for subgraph tensorrt_0
  Finished building TensorRT engine for subgraph tensorrt_0


*****************************
Additional Flags for TensorRT
*****************************

DLR provides several runtime flags to configure the TensorRT components of your optimized model.
These flags are all configured through environment variables.

The examples will use the following test script ``run.py``.
  
.. code-block:: python

  import dlr
  import numpy as np
  import time

  model = dlr.DLRModel('my_compiled_model/', 'gpu', 0)
  x = np.random.rand(1, 3, 224, 224)
  # Warmup
  model.run(x)

  times = []
  for i in range(100):
    start = time.time()
    model.run(x)
    times.append(time.time() - start)
  print('Latency:', 1000.0 * np.mean(times), 'ms')

Example output

.. code-block:: bash

  $ python3 run.py

  Building new TensorRT engine for subgraph tensorrt_0
  Finished building TensorRT engine for subgraph tensorrt_0
  Latency: 3.320300579071045 ms

Automatic FP16 Conversion
-------------------------

Environment variable ``TVM_TENSORRT_USE_FP16=1`` can be set to automatically convert the TensorRT
components of your model to 16-bit floating point precision. This can greatly increase performance,
but may cause some slight loss in the model accuracy.

.. code-block:: bash

  $ TVM_TENSORRT_USE_FP16=1 python3 run.py

  Building new TensorRT engine for subgraph tensorrt_0
  Finished building TensorRT engine for subgraph tensorrt_0
  Latency: 1.7122554779052734 ms


Caching TensorRT Engines
------------------------

During the first inference, DLR will invoke the TensorRT API to build an engine. This can be time consuming, so you can set ``TVM_TENSORRT_CACHE_DIR``
to point to a directory to save these built engines to on the disk. The next time you load the model and give it the same directory,
DLR will load the already built engines to avoid the long warmup time.

.. code-block:: bash

  $ TVM_TENSORRT_CACHE_DIR=. python3 run.py

  Building new TensorRT engine for subgraph tensorrt_0
  Caching TensorRT engine to ./8030730458607885728.plan
  Finished building TensorRT engine for subgraph tensorrt_0
  Latency: 4.380748271942139 ms

  $ TVM_TENSORRT_CACHE_DIR=. python3 run.py

  Loading cached TensorRT engine from ./8030730458607885728.plan
  Latency: 4.414560794830322 ms

Changing the TensorRT Workspace Size
------------------------------------

TensorRT has a paramter to configure the maximum amount of scratch space that each layer in the model can use.
It is generally best to use the highest value which does not cause you to run out of memory.
Neo will automatically set the max workspace size to 256 megabytes for Jetson Nano and Jetson TX1 targets, and 1 gigabyte for all other NVIDIA GPU targets.
You can use ``TVM_TENSORRT_MAX_WORKSPACE_SIZE`` to override this by specifying the workspace size in bytes you would like to use.

.. code-block:: bash

  $ TVM_TENSORRT_MAX_WORKSPACE_SIZE=2147483647 python3 run.py
