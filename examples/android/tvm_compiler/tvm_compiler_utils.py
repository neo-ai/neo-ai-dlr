#!/usr/bin/env python3

import os
import sys
import tvm
from tvm import relay
from tvm.contrib import ndk

sys.setrecursionlimit(1031)
print("sys.getrecursionlimit()", sys.getrecursionlimit())


def tvm_compile(func, params, arch, dlr_model_name):
  ###arch x86_64
  if arch == 'x86_64':
    target = "llvm -model=N3350 -target=x86_64-linux-android -mattr=+ssse3,+sse4.2"
    sysroot="/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot"
    toolchain="/opt/android-ndk/toolchains/x86_64-4.9/prebuilt/linux-x86_64"
    os.environ['TVM_NDK_CC'] = "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android28-clang++"
  ###arch x86 i686
  elif arch == 'x86':
    target = "llvm -model=x5-Z8350 -target=i686-linux-android -mattr=+ssse3"
    sysroot="/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot"
    toolchain="/opt/android-ndk/toolchains/x86-4.9/prebuilt/linux-x86_64"
    os.environ['TVM_NDK_CC'] = "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android21-clang++"

  ###arch arm64 aarch64
  elif arch == 'arm64-v8a':
    target = "llvm -device=arm_cpu -model=SM8150 -target=aarch64-linux-android"
    sysroot="/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot"
    toolchain="/opt/android-ndk/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64"
    os.environ['TVM_NDK_CC'] = "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++"

  ###arch armv7
  ## More info on armv7 hard/soft abi for Android https://android.googlesource.com/platform/ndk/+/master/docs/HardFloatAbi.md
  elif arch == 'armeabi-v7a':
    target = "llvm -device=arm_cpu -model=MSM8940 -target=armv7a-linux-androideabi -mfloat-abi=soft -mattr=+neon,+thumb-mode"
    sysroot="/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot"
    toolchain="/opt/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64"
    os.environ['TVM_NDK_CC'] = "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi21-clang++"
  else:
    print("Valid arch: arm64-v8a, armeabi-v7a, x86_64, x86")
    return

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
  options=["-shared", "-fPIC", "--sysroot", sysroot, "--gcc-toolchain="+toolchain, "-static-libstdc++"]
  lib.export_library(path_lib, ndk.create_shared, options=options)

  print("export_library done")

  with open(out_folder + "model.json", "w") as fo:
    fo.write(graph)
  with open(out_folder + "model.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

  print("Files saved to", out_folder)
