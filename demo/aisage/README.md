# Acer aiSage Demo

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

### TVM runtime and gluoncv
Install TVM runtime
```
sudo easy_install3 tvm-0.6.dev0-py3.5-linux-aarch64.egg
```

Install gluoncv
```
sudo pip3 install gluoncv
```

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
