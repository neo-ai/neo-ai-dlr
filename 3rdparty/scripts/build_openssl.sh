#!/bin/bash

if [ -z "$ANDROID_NDK_HOME" ]; then
  echo "ANDROID_NDK_HOME variable not set"
  local_prop=$2/../../../../../local.properties

  if [ -f $local_prop ]; then
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
if [ "$3" = "x86_64" ]; then 
  export TARGET_HOST=x86_64-linux-android
  export BARCH=x86_64
elif [ "$3" = "x86" ]; then 
  export TARGET_HOST=i686-linux-android
  export BARCH=x86
elif [ "$3" = "arm64-v8a" ]; then 
  export TARGET_HOST=aarch64-linux-android
  export BARCH=arm64
else 
  export TARGET_HOST=armv7a-linux-androideabi
  export BARCH=arm
fi
mkdir -p $1/build

$1/Configure android-$BARCH \
 -D__ANDROID_API__=$MIN_SDK_VERSION \
 --prefix=$1/build/$3

make -j4
make install
make clean
