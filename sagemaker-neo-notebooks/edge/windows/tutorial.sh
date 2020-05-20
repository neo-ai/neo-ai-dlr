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


# build dlr
git clone --recursive https://github.com/neo-ai/neo-ai-dlr
cd neo-ai-dlr
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
export DLR_LIB=$(pwd)/lib
