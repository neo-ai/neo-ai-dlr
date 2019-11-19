"""
All userspace code authored by Xilinx is released under the following license:
Copyright (C) 2019 Xilinx, Inc
Licensed under the Apache License, Version 2.0 (the "License"). You may
not use this file except in compliance with the License. A copy of the
License is located at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

"""
Registration of Xilinx Vitis-AI fused acceleration operation for aceleration of 
convolutional neural networks


This Vitis-AI acceleration exploits DPU hardware accelerator built for 
following evaluation boards:
- Ultra96: https://www.xilinx.com/products/boards-and-kits/1-vad4rl.html
- ZCU104: https://www.xilinx.com/products/boards-and-kits/zcu104.html
- ZCU102: https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html
- Zedboard: https://www.xilinx.com/products/boards-and-kits/1-8dyf-11.html

More information about DPU and DNNDK can be found in the user guides:
https://www.xilinx.com/support/documentation/ip_documentation/dpu/v3_0/pg338-dpu.pdf
https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf
"""

import tvm
import warnings
import numpy as np

try:
    from dnndk import n2cube
except:
    warnings.warn("Could not import dnndk n2cube")


@tvm.register_func("tvm.accel.accel_fused")
def accel_fused(kernel_name, input_name, output_name, 
    layout, out, *ins):

    # Attach to DPU driver and prepare for running
    n2cube.dpuOpen()
    
    # Create DPU Kernels
    kernel = n2cube.dpuLoadKernel(kernel_name)

    # Create DPU Tasks for kernel
    task = n2cube.dpuCreateTask(kernel, 0)

    # Load image to DPU
    X = ins[0].asnumpy().reshape((-1))
    n2cube.dpuSetInputTensorInHWCFP32(task, input_name, X, len(X))

    # Model run on DPU """
    n2cube.dpuRunTask(task)
    
    # Get the output tensor size 
    size = n2cube.dpuGetOutputTensorSize(task, output_name)
    address = n2cube.dpuGetOutputTensorAddress(task, output_name)

    value = [0 for i in range(size)]

    # Get the output tensor data 
    n2cube.dpuGetTensorData(address, value, size)
    scale = n2cube.dpuGetOutputTensorScale(task, output_name, idx=0)
    
    value = np.array(value).astype(np.float32) * float(scale)
    
    value_shape = tuple(out.shape) if layout == 'NHWC' else  \
        (out.shape[0], out.shape[2], out.shape[3], out.shape[1])
    value = np.reshape(value, value_shape)

    # DPU output is in NHWC
    if layout == 'NCHW':
        value = np.transpose(value,(0,3,1,2))
    
    tvm.nd.array(value).copyto(out)
