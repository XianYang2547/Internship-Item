<a align="left">
  <a href="https://github.com/XianYang2547">
    <img src="https://img.shields.io/badge/Author-@XianYang-000000.svg?logo=GitHub" alt="GitHub">
  </a>
</a>

## Directory Structure
```
.
├── assets             	   # test data
│ ├── rtsp             
│ ├── test.jpg
│ ├── test.avi
│ └── test.svo
├── C_code          	   # c++ TODO
│ ├── bytetrack
│ ├── CMakeLists.txt
│ ├── main.cpp
│ ├── segment
│ └── xytools
├── Infer_Python       		
│ ├── demo.py          		# demo
│ ├── *.py             		# ...
│ ├── script           		# socket save script
│ └── xy               		# 
├── models             		
│ ├── get_trt.py       		# py
│ ├── get_trt.cpp      		# cpp
│ └── get_engine       		# two
├── output             		
│ └── app.log				# run log file
├── test			   		
├── tools
│ ├── biaoding_lidar2camera # manual calib
│ ├── camera_aarch			# zed arrch
│ ├── camera_x86_64			# zed x86
│ ├── lidar					# AT128 lidar ros&sdk
│ └── socket				# java jar package
├── download.sh				# down script
├── README.md
└── requirements.txt
```
## On ubuntu22.04
### 0. Clone it
```bash
git clone http://172.16.10.64/xianyang/Road_Assets.git
cd Road_Assets
```

### 1. Environmental Installation
```bash
pip install -r requirements.txt
cat requirements.txt
```

### 2. Model Transform
- 2.1 download model
```bash
bash download.sh
```
- 2.2 use py script
```bash
python ./model_files/get_trt.py --onnx model.onnx --engine model.plan --fp16 True
```
- 2.3 use c++ script
```bash
g++ -o get_engine get_trt.cpp -I/home/xianyang/Documents/TensorRT-8.6.1.6/include -L/home/xianyang/Documents/TensorRT-8.6.1.6/lib -lnvinfer -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcudart -lnvonnxparser
./get_engine your_onnx_path your_engine_path 
```

### 3. Run Example
 - for image/avi/svo/directory
```bash
python Infer_Python/demo.py --path [your file or directory]
```
 - for camera 
```bash
cd tools/camera_x86_64
cat README.md
```
```bash
python Infer_Python/zed_infer.py
```
 - for lidar
```bash
cd tools/lidar
bash start_lidar.sh
```
```bash
python Infer_Python/lidar_infer.py
```
 - for camera、lidar、socket（java），run java -jar *.jar support start、pause、resume、stop
```bash
cd tools/socket
java -jar *.jar
```
```bash
python Infer_Python/lidar_infer_socket.py
```

****

## 🎖 贡献者
<a href="https://github.com/XianYang2547/Internship-Item/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=XianYang2547/Internship-Item" />
</a>
