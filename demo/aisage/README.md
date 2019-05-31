# Acer aiSage Demo

Table of Contents
=================

   * [Acer aiSage Demo](#acer-aisage-demo)
   * [Table of Contents](#table-of-contents)
      * [Installation](#installation)
         * [pip](#pip)
         * [Tensorflow](#tensorflow)
         * [TVM runtime and OpenCV](#tvm-runtime-and-opencv)
         * [DLR](#dlr)
      * [Run Inference](#run-inference)
         * [ssd_mobilenet_v1_0.75_depth_quantized_coco](#ssd_mobilenet_v1_075_depth_quantized_coco)
         * [YOLOv3](#yolov3)
         * [Gluoncv yolo3_darknet53_voc](#gluoncv-yolo3_darknet53_voc)
         * [Gluoncv ssd_512_mobilenet1.0_voc](#gluoncv-ssd_512_mobilenet10_voc)
         * [MXNet SSD Mobilenet 512 voc](#mxnet-ssd-mobilenet-512-voc)
         * [MXNet SSD Resnet50 512 voc](#mxnet-ssd-resnet50-512-voc)

## Installation

### pip
Install pip3
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
```

### Tensorflow
Install Tensorflow
```
curl -O https://github.com/neo-ai/neo-ai-dlr/blob/demo-aisage/demo/aisage/tensorflow-1.13.1-cp35-none-linux_aarch64.whl
sudo pip3 install --upgrade tensorflow-1.13.1-cp35-none-linux_aarch64.whl
```

### TVM runtime and OpenCV
Install TVM runtime
```
sudo easy_install3 tvm-0.6.dev0-py3.5-linux-aarch64.egg
```

Install OpenCV as described below

https://docs.opencv.org/4.0.1/d7/d9f/tutorial_linux_install.html


### DLR
Install DLR
```
curl -O https://github.com/neo-ai/neo-ai-dlr/blob/demo-aisage/demo/aisage/dlr-1.0-py2.py3-none-any.whl
sudo pip3 install --upgrade dlr-1.0-py2.py3-none-any.whl
```

## Run Inference

### ssd_mobilenet_v1_0.75_depth_quantized_coco
Edit `run-ssd.py` file and make sure that the following line is uncommented
```
model_path, input_tensor_name = "models/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite", "normalized_input_image_tensor"
```
Run the inference
```
python3 run-ssd.py
```
The script runs inference 10 times and gives you the result (boxes,classes,scores) and time stats
```
model: ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite
(1, 300, 300, 3) uint8
Memory RSS: 30,089,216
Memory RSS: 170,946,560
input names: ['normalized_input_image_tensor']
output names: ['TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1', 'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3']
dryrun...
Memory RSS: 177,942,528
1 m.run...
Memory RSS: 177,942,528
m.run done, duration 131 ms
dogs.jpg - found objects:
   18 dog 0.6875 [0.0225764  0.02660308 0.86825544 0.3236617 ]
   18 dog 0.64453125 [0.09096082 0.2967648  0.92600816 0.9637793 ]
...
10 m.run...
Memory RSS: 178,184,192
m.run done, duration 132 ms
dogs.jpg - found objects:
   18 dog 0.6875 [0.0225764  0.02660308 0.86825544 0.3236617 ]
   18 dog 0.64453125 [0.09096082 0.2967648  0.92600816 0.9637793 ]
Avg: 131.7 ms, Median 132.0 ms (stddev: 0.48304589153964794)
Memory RSS: 178,184,192
```

### YOLOv3
Set Swap memory size to minimum 3GB
```
free -h
```
Copy frozen graph file `yolov3.pb` to models folder
```
cd models
curl -O https://s3.us-east-2.amazonaws.com/dlc-models/yolov3/yolov3.pb
cd ..
```
Run the inference
```
python3 run-yolov3.py
```
The script runs inference 10 times and gives you the result (boxes,scores,classes) and time stats
```
model: models/yolov3.pb
(1, 416, 416, 3) float32
Memory RSS: 57,655,296
Memory RSS: 586,051,584
input names: ['import/input_data:0']
output names: ['import/concat_10:0', 'import/concat_11:0', 'import/concat_12:0']
dryrun...
Memory RSS: 1,644,531,712
1 m.run...
Memory RSS: 1,710,432,256
m.run done, duration 7,396 ms
dog.jpg - objects:
   1 bicycle 0.98989916 [ 87.91115 166.8269  427.18823 576.592  ]
   7 truck 0.93969005 [255.95006   62.862793 375.19208  120.23923 ]
   16 dog 0.9979028 [ 66.49128 161.10391 173.2828  392.7958 ]
...
10 m.run...
Memory RSS: 1,836,171,264
m.run done, duration 6,993 ms
dog.jpg - objects:
   1 bicycle 0.98989916 [ 87.91115 166.8269  427.18823 576.592  ]
   7 truck 0.93969005 [255.95006   62.862793 375.19208  120.23923 ]
   16 dog 0.9979028 [ 66.49128 161.10391 173.2828  392.7958 ]
Avg: 7,246.7 ms, Median 7,261.5 ms (stddev: 459.6477032588231)
Memory RSS: 1,836,998,656
```

### Gluoncv yolo3_darknet53_voc
Download compiled model from s3 bucket
```
cd models

mkdir yolov3_darknet53

cd yolov3_darknet53

curl -O https://s3.us-east-2.amazonaws.com/dlc-models/demo_yolo_v3_darknet_300_acer/deploy_param.params

curl -O https://s3.us-east-2.amazonaws.com/dlc-models/demo_yolo_v3_darknet_300_acer/deploy_graph.json

curl -O https://s3.us-east-2.amazonaws.com/dlc-models/demo_yolo_v3_darknet_300_acer/deploy_lib.so

cd ../..
```

Run the inference
```
python3 run_yolo_gluoncv.py
```
The script will run the model 10 times and dumps the result to console:
```
Inference time: 1,284 ms
Inference time: 1,274 ms
Inference time: 1,284 ms
Inference time: 1,263 ms
Inference time: 1,273 ms
Inference time: 1,262 ms
Inference time: 1,342 ms
Inference time: 1,294 ms
Inference time: 1,271 ms
Inference time: 1,285 ms
0 14 person 0.9947595 [ 16.338623 121.93883   57.51333  209.91515 ]
1 6 car 0.9909543 [149.0418  128.55173 216.31621 169.59064]
2 1 bicycle 0.9865054 [175.475   153.96129 282.6286  238.53874]
3 14 person 0.95648724 [117.49012 115.61078 148.5764  191.71698]
4 14 person 0.9378013 [195.87904 123.14607 265.96475 228.11818]
5 14 person 0.72334594 [ 78.89762  124.670456  87.09647  158.53853 ]
6 14 person 0.53352076 [ 46.620457 125.701904  59.702282 159.13812 ]
```
### Gluoncv ssd_512_mobilenet1.0_voc
Download compiled model from s3 bucket
```
cd models

mkdir ssd_mobilenet1.0

cd ssd_mobilenet1.0

curl -O https://s3.us-east-2.amazonaws.com/dlc-models/demo_ssd_mobilenet1.0_300_acer/deploy_param.params

curl -O https://s3.us-east-2.amazonaws.com/dlc-models/demo_ssd_mobilenet1.0_300_acer/deploy_graph.json

curl -O https://s3.us-east-2.amazonaws.com/dlc-models/demo_ssd_mobilenet1.0_300_acer/deploy_lib.so

cd ../..
```

Run the inference
```
python3 run_ssd_gluoncv.py
```
The script has 5 warmup runs and 10 test runs, output as follows:
```
warm up..
cost per image: 0.4823s
test..
cost per image: 0.4712s
```

### MXNet SSD Mobilenet 512 voc
Download compiled mxnet-ssd-mobilenet-512 model from s3
```
cd models
mkdir mxnet-ssd-mobilenet-512
cd mxnet-ssd-mobilenet-512
curl -O https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-mobilenet-512/model.params
curl -O https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-mobilenet-512/model.json
curl -O https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-mobilenet-512/model.so
cd ../..
```
Run the inference
```
python3 run-mxnet-ssd-mobilenet-512.py
```
Script warms up the model and runs the inference 10 times. The output:
```
models/mxnet-ssd-mobilenet-512/model.so
Warming up...
Running...
time: 1391 ms
time: 686 ms
time: 672 ms
time: 641 ms
time: 660 ms
time: 700 ms
time: 703 ms
time: 659 ms
time: 666 ms
time: 687 ms
1  car [6.         0.9542924  0.6018397  0.13442558 0.89551395 0.29527432]
2  dog [11.          0.8844297   0.14165549  0.38350004  0.40806746  0.9418139 ]
```
To display the image with boundary boxes uncomment two last lines in the script

### MXNet SSD Resnet50 512 voc
Download compiled mxnet-ssd-resnet50-512 model from s3
```
cd models
mkdir mxnet-ssd-resnet50-512
cd mxnet-ssd-resnet50-512
curl -O https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-resnet50-512/model.params
curl -O https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-resnet50-512/model.json
curl -O https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-resnet50-512/model.so
cd ../..
```
Run the inference
```
python3 run-mxnet-ssd-resnet50-512.py
```
Script warms up the model and runs the inference 10 times. The output:
```
models/mxnet-ssd-resnet50-512/model.so
Warming up...
Running...
time: 6694 ms
time: 3300 ms
time: 3336 ms
time: 3473 ms
time: 3376 ms
time: 3219 ms
time: 3293 ms
time: 3299 ms
time: 3182 ms
time: 3330 ms
1 car [6.         0.9989802  0.6128727  0.13521719 0.8909892  0.294108  ]
2 dog [11.          0.91762155  0.168013    0.33782458  0.39899638  0.9333589 ]
3 bicycle [1.         0.8148663  0.18910009 0.2178039  0.7376316  0.7913436 ]
```
To display the image with boundary boxes uncomment two last lines in the script
