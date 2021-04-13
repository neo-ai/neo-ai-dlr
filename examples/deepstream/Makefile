CUDA_PATH=/usr/local/cuda
TENSORRT_HEADER_PATH=/usr/include/x86_64-linux-gnu
TENSORRT_LIB_PATH=/usr/lib/x86_64-linux-gnu
DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream/sources
ifeq ($(CUDA_PATH),)
  $(error "CUDA_PATH is not set")
endif
ifeq ($(TENSORRT_HEADER_PATH),)
  $(error "TENSORRT_HEADER_PATH is not set")
endif
ifeq ($(TENSORRT_LIB_PATH),)
  $(error "TENSORRT_LIB_PATH is not set")
endif
ifeq ($(DEEPSTREAM_PATH),)
  $(error "DEEPSTREAM_PATH is not set")
endif
CC:= g++

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I$(DEEPSTREAM_PATH)/includes \
         -I$(CUDA_PATH)/include \
		 -I$(TENSORRT_HEADER_PATH) \
		 -I../../3rdparty/tvm/3rdparty/dlpack/include

LIBS:= -ldl -lstdc++fs \
       -L$(TENSORRT_LIB_PATH)/lib -lnvinfer_plugin -lnvinfer -lnvparsers \
	   -L$(CUDA_PATH)/lib64 -lcudart -lcublas
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
SRCFILES:= neodlr_plugin.cc nvdsinfer_custom_impl_neodlr.cc
TARGET_LIB:= libnvdsinfer_custom_impl_neodlr.so

TARGET_OBJS:= $(SRCFILES:.cc=.o)

all: $(TARGET_LIB)

%.o: %.cc $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
