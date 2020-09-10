# yolov3_traffic_violation_tracking
This application is an automated system of traffic monitoring to detect violations. As of now, helmet violations detection is supported, but in future I am planning to add wrong side driving detection. The program automatically generates traffic violation tickets (or challans) based on registration details added to the database. Demonstration of the application along with the explanation of the source code is shown in [this video](https://github.com/sriramcu/yolov3_traffic_violation_tracking/blob/master/demo_full.mp4).  
The above video will be documented here in the near future.  
  
This project is based on three currently existing git repos:  
1. [BlcaKHat/yolov3-Helmet-Detection](https://github.com/BlcaKHat/yolov3-Helmet-Detection)
2. [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
3. [marcbelmont/deep-license-plate-recognition](https://github.com/marcbelmont/deep-license-plate-recognition)  

Each of the above repos have been modified to suit my application and can be found in their respective directories with the original README and LICENSE. Since the first repo  is licensed under GPL, this repo will also be licensed under GPL. Refer to 

## Setup
Follow these instructions:  
1. Initial installation  
```console  
$ git clone https://github.com/sriramcu/yolov3_traffic_violation_tracking
$ cd yolov3_traffic_violation_tracking
$ pip install -r requirements.txt
$ sudo apt update 
$ sudo apt install libqt5x11extras5
$ sudo apt install libgl1-mesa-glx
```

