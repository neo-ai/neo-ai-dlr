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

### DLR
Install DLR
```
curl -O ???
sudo pip3 instal --upgrade ???
```

## Run Inference

### ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite
Edit `run-dlr.py` file and make sure that the following line is uncommented
```
model_path, input_tensor_name = "models/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite", "normalized_input_image_tensor"
```
Run the inference
```
python3 run-dlr.py
```
The script runs inference 10 times and gives you the result (boxes) and time stats
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
