#!/bin/bash

if [ -z "$ANDROID_NDK_HOME" ]; then
  echo "ANDROID_NDK_HOME variable empty"
  local_prop=$2/../../../../../local.properties

  if [ -f $local_prop ]; then
    echo "file exist"
    ndk_val=`cut -d '=' -f 2 $local_prop`
    export ANDROID_NDK_HOME="$ndk_val/ndk/20.1.5948944"
    echo $ANDROID_NDK_HOME
  else
    echo "android local.properties file not exist"
  fi
fi

export HOST_TAG=linux-x86_64
export MIN_SDK_VERSION=21
export TOOLCHAIN=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$HOST_TAG
PATH=$TOOLCHAIN/bin:$PATH
export NDK=$ANDROID_NDK_HOME
export TOOLCHAIN=$NDK/toolchains/llvm/prebuilt/$HOST_TAG
export AR=$TOOLCHAIN/bin/aarch64-linux-android-ar
export AS=$TOOLCHAIN/bin/aarch64-linux-android-as
export CC=$TOOLCHAIN/bin/aarch64-linux-android21-clang
export CXX=$TOOLCHAIN/bin/aarch64-linux-android21-clang++
export LD=$TOOLCHAIN/bin/aarch64-linux-android-ld
export RANLIB=$TOOLCHAIN/bin/aarch64-linux-android-ranlib
export STRIP=$TOOLCHAIN/bin/aarch64-linux-android-strip
export TARGET_HOST=x86_64-linux-android
mkdir -p $1/build

$1/Configure android-x86_64 \
 -D__ANDROID_API__=$MIN_SDK_VERSION \
 --prefix=$1/build/x86_64 

make -j4
make install

