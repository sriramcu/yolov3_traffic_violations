# yolov3_traffic_violation_tracking
This application is an automated system of traffic monitoring to detect violations. As of now, helmet violations detection is supported, but in future I am planning to add wrong side driving detection. The program automatically generates traffic violation tickets (or challans) based on registration details added to the database. Demonstration of the application along with the explanation of the source code is shown in [this video](https://github.com/sriramcu/yolov3_traffic_violation_tracking/blob/master/demo_full.mp4).  
The above video will be documented here in the near future.  
  
This project is based on three currently existing git repos:  
1. [BlcaKHat/yolov3-Helmet-Detection](https://github.com/BlcaKHat/yolov3-Helmet-Detection)
2. [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
3. [marcbelmont/deep-license-plate-recognition](https://github.com/marcbelmont/deep-license-plate-recognition)  

Each of the above repos have been modified to suit my application and can be found in their respective directories with the original README and LICENSE. Since the first repo  is licensed under GPL v3.0, this repo will also be licensed under GPL v3.0. Refer to [LICENSE](https://github.com/sriramcu/yolov3_traffic_violation_tracking/blob/master/LICENSE) file for details.

## Setup
Follow these instructions:  
I) Initial installation  
```console  
$ git clone https://github.com/sriramcu/yolov3_traffic_violation_tracking
$ cd yolov3_traffic_violation_tracking
$ pip install -r requirements.txt
$ sudo apt update 
$ sudo apt install libqt5x11extras5
$ sudo apt install libgl1-mesa-glx
```  
II) Downloading Large Files  
Since these files exceed 100 MB, you must visit the links given below and add the files in the proper directory.**The program will fail if these files aren't placed in the correct directory.**  
A pull request for performing this step using the console would be appreciated.

1.  [yolov3.weights](https://drive.google.com/file/d/16XNa9Zt5GfgeCNW0fl8Hfx9ZaPQ2OEtt/view?usp=sharing). Put this file in the keras-yolo3/ subdirectory.  
2. [core](https://drive.google.com/file/d/17Uu7X9-MI0e0SetrV2ZReHOk1buWPfMH/view?usp=sharing). Put this file in the yolov3_Helmet_Detection/ subdirectory.  
3. [yolov3-obj_2400.weights](https://drive.google.com/file/d/16Pr_4FbOkoktDDE8rZpB8b2bh-GYouJl/view?usp=sharing). Put this file in the yolov3_Helmet_Detection/ subdirectory.

III) Set up API key for ANPR
Get your ANPR API key from [here](https://platerecognizer.com/?utm_source=github&utm_medium=website). Name this file as api_key.txt and place it in the current folder(yolov3_traffic_violation_tracking).


## Usage
```console
$ python3 gui.py
```
Above command launches the GUI from where you can run the program.  

That's all for now. But much more is still to come in the future! Stay tuned.  



