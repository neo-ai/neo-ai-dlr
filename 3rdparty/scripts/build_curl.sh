#!/bin/bash

if [ -z "$ANDROID_NDK_HOME" ]; then
  echo "ANDROID_NDK_HOME variable not set"
  local_prop=$2/../../../../../local.properties

  if [ -f $local_prop ]; then
    echo "local property file exist"
    ndk_val=`cut -d '=' -f 2 $local_prop`
    export ANDROID_NDK_HOME="$ndk_val/ndk/20.1.5948944"
    echo $ANDROID_NDK_HOME
  else
    echo "android local.properties file not exist"
  fi
fi

cd $1 
./buildconf
export MIN_SDK_VERSION=21
export LIBS="-lssl -lcrypto"
export CFLAGS="-mthumb"
export HOST_TAG=linux-x86_64
if [ "$3" = "x86_64" ]; then
  export TARGET_HOST=x86_64-linux-android
elif [ "$3" = "x86" ]; then
  export TARGET_HOST=i686-linux-android
elif [ "$3" = "arm64-v8a" ]; then
  export TARGET_HOST=aarch64-linux-android
else
  export TARGET_HOST=armv7a-linux-androideabi
fi
export TOOLCHAIN=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$HOST_TAG
PATH=$TOOLCHAIN/bin:$PATH
export AR=$TOOLCHAIN/bin/$TARGET_HOST-ar
export AS=$TOOLCHAIN/bin/$TARGET_HOST-as
export CC=$TOOLCHAIN/bin/$TARGET_HOST$MIN_SDK_VERSION-clang
export CXX=$TOOLCHAIN/bin/$TARGET_HOST$MIN_SDK_VERSION-clang++
export LD=$TOOLCHAIN/bin/$TARGET_HOST-ld
export RANLIB=$TOOLCHAIN/bin/$TARGET_HOST-ranlib
export STRIP=$TOOLCHAIN/bin/$TARGET_HOST-strip
export SSL_DIR=$1/../openssl/build/$3
export LDFLAGS="-L$SSL_DIR/lib"

if [ "$3" = "armeabi-v7a" ]; then 
  export AR=$TOOLCHAIN/bin/aarch64-linux-android-ar
  export AS=$TOOLCHAIN/bin/aarch64-linux-android-as
  export LD=$TOOLCHAIN/bin/aarch64-linux-android-ld
  export RANLIB=$TOOLCHAIN/bin/aarch64-linux-android-ranlib
  export STRIP=$TOOLCHAIN/bin/aarch64-linux-android-strip
fi

mkdir -p $1/build
$1/configure --host=$TARGET_HOST \
            --target=$TARGET_HOST \
            --prefix=$1/build/$3 \
            --with-ssl=$SSL_DIR \
            --without-librtmp \
            --enable-hidden-symbols \
            --without-libidn \
            --without-librtmp \
            --without-zlib \
            --disable-dict \
            --disable-file \
            --disable-ftp \
            --disable-ftps \
            --disable-gopher \
            --disable-imap \
            --disable-imaps \
            --disable-pop3 \
            --disable-pop3s \
            --disable-smb \
            --disable-smbs \
            --disable-smtp \
            --disable-smtps \
            --disable-telnet \
            --disable-tftp \
            --enable-threaded-resolver \
            --enable-libgcc \
            --enable-ipv6 \
            --enable-manual \
            --enable-tls-srp \
            --enable-crypto-auth \
            --enable-verbose \
            --enable-ipv6 \
            --enable-static \
            --enable-shared

make -j4 
make install
make clean
