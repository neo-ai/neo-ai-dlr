#  build depedencies
git clone https://github.com/nlohmann/json && \
    cd json && \
    cmake . && \
    make && \
    make install


#  build AWS SDK depedencies
git clone https://github.com/aws/aws-sdk-cpp.git
cd aws-sdk-cpp
mkdir build && cd build
cmake .. \
  -DBUILD_ONLY="s3" \
  -DBUILD_SHARED_LIBS=OFF \
  -DENABLE_UNITY_BUILD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/aws-install

make -j$(nproc)
make install

# install xtensor
https://xtensor.readthedocs.io/en/latest/installation.html
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=path_to_prefix ..
make install


# build dlr
git clone --recursive https://github.com/neo-ai/neo-ai-dlr
cd neo-ai-dlr
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
export DLR_LIB=$(pwd)/lib

# link file
ln -s $DLR_LIB/libdlr.so
ln -s $DLR_LIB/libdlr.dylib

# for windows
C:\Program Files (x86)\aws-cpp-sdk-all\lib\cmake
cmake .. -DBUILD_ONLY="s3" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/aws-install
cmake .. -DCMAKE_PREFIX_PATH="C:\Program Files (x86)\aws-cpp-sdk-all\lib\;C:\Program Files (x86)\aws-cpp-sdk-all\lib\cmake" -DBUILD_SHARED_LIBS=ON
cmake ../ -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/aws-cpp-sdk-all/lib/;C:/Program Files (x86)/aws-cpp-sdk-all/lib/cmake" -DBUILD_SHARED_LIBS=ON
