FROM ubuntu:16.04

RUN apt update

RUN apt install -y build-essential
RUN apt install -y llvm-6.0 clang-6.0
RUN apt install -y git wget curl ca-certificates openssl vim zip
RUN apt install -y python3 python3-dev python3-distutils
RUN apt install -y libedit-dev libxml2-dev antlr4

RUN wget https://cmake.org/files/v3.22/cmake-3.22.0-linux-x86_64.sh \
    && bash cmake-3.22.0-linux-x86_64.sh --skip-license --prefix=/usr/local

RUN curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py

RUN pip3 install -U pip setuptools wheel
RUN pip3 install numpy decorator attrs antlr4-python3-runtime

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y cuda-nvcc-10-0 cuda-libraries-10-0 cuda-nvrtc-dev-10-0 cuda-cudart-dev-10-0 cuda-runtime-10-0
ENV PATH="${PATH}:/usr/local/cuda-10.0/bin"

WORKDIR /root
RUN git clone --recursive https://github.com/apache/incubator-tvm.git tvm
WORKDIR /root/tvm/build
RUN cmake -DUSE_LLVM=ON -DUSE_ANTLR=ON -DUSE_CUDA=ON .. && make -j$(nproc)

WORKDIR /root/tvm/python
RUN python3 setup.py install
WORKDIR /root/tvm/topi/python
RUN python3 setup.py install

WORKDIR /root
RUN pip3 install tensorflow==1.15 keras keras-applications
RUN pip3 install mxnet gluoncv

WORKDIR /root/ec2_compilation_container
COPY tvm_ec2_compiler_utils.py compile_keras.py compile_tensorflow.py compile_gluoncv.py ./
