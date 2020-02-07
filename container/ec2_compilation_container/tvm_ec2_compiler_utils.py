#!/usr/bin/env python3

import os
import sys
import tvm
from tvm import relay
from tvm.contrib import ndk

sys.setrecursionlimit(1031)
print("sys.getrecursionlimit()", sys.getrecursionlimit())


def tvm_compile(func, params, arch, dlr_model_name):
  gpu_code = None
  ###arch c4 avx2
  if arch in ['c4', 'm4']:
    target = "llvm -mcpu=core-avx2"
  ###arch c5 avx512
  elif arch in ['c5', 'm5']:
    target = "llvm -mcpu=skylake-avx512"
  elif arch in ['p3', 'ml_p3']:
    target = "cuda"
    gpu_code = "sm_70"
  elif arch in ['p2', 'ml_p2']:
    target = "cuda"
    gpu_code = "sm_37"
  ###arch lambda ssse3,sse4.2,avx
  elif arch == 'lambda':
    target = "llvm -mcpu=ivybridge"
  else:
    print("Valid arch: c4, m4, c5, m5, lambda")
    return

  if gpu_code is not None:
    #set cuda arch before relay.build
    from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
    set_cuda_target_arch(gpu_code)
    print("gpu_code:", gpu_code)

  print('target:', target)
  print("Compiling...")

  with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(func, target, params=params)

  print("Compilation done")
  print("lib type_key: ", lib.type_key)

  print("Saving files")
  out_folder = arch + "/" + dlr_model_name + "/"
  os.makedirs(out_folder, exist_ok=True)
  # save the graph, lib and params into separate files
  path_lib = out_folder + "model.so"
  lib.export_library(path_lib)

  print("export_library done")

  with open(out_folder + "model.json", "w") as fo:
    fo.write(graph)
  with open(out_folder + "model.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

  print("Files saved to", out_folder)
