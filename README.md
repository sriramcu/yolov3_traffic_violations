# Yolov3 Traffic Violations - Helmet Detection and ANPR

This application is an automated system of traffic monitoring to detect helmet violations. The system for 
detecting wrong side driving violations is implemented in YOLOv4 in 
[this repo](https://github.com/sriramcu/yolov4_wrong_side_driving_detection). The program 
automatically generates traffic violation tickets (or challans) based on registration details added to the 
database. 

## Acknowledgements

This project is based on three currently existing git repos:  
1. [BlcaKHat/yolov3-Helmet-Detection](https://github.com/BlcaKHat/yolov3-Helmet-Detection)
2. [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
3. [marcbelmont/deep-license-plate-recognition](https://github.com/marcbelmont/deep-license-plate-recognition)  

Each of the above repos have been modified to suit my application and can be found in their respective 
directories with the original README and LICENSE. 

## High level overview

[BlcaKHat/yolov3-Helmet-Detection](https://github.com/BlcaKHat/yolov3-Helmet-Detection) already takes care of 
detecting and counting number of people wearing helmets in an image. 
[qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3) was trained on a dataset containing people and 
bikes but not helmets. 

* Our project aims to combine the two functionalities to avoid detecting violations for 
pedestrians not wearing helmets or scooters that are parked with no helmet. 
* It also detects violations where a helmet is present but not worn, for instance hanging on the side of a bike.
* The program detects a violation only if a helmet's bounding box is inside that of a person's and the person's 
  bounding box is inside that of a bike's.
* These violations are cropped out of the overall video frame to the combined dimension of the bike's and the 
  person's bounding boxes with some buffer and stored inside cropped_images folder inside keras_yolo3 folder.
* The ANPR program is run on these images and stored into violations folder inside keras_yolo3 folder.
* A challan is generated for each violation by referencing the vehicles.db sqlite3 database stored in 
  the root directory of the project using the license plate of the vehicle and is stored in challans 
  folder in the root directory of the project.
* Before running the helmet detection program, vehicle data is assumed to have been entered via the GUI, i.e. 
  vehicle license plates, name and address of the owner (3 columns).

### Changes made to submodules referenced in the Acknowledgments Section

* The deep_license_plate_recognition module has minimal changes. 
* The yolov3_Helmet_Detection folder contains moderate changes, such as minor tweaks in the Helmet_detection_YOLOv3.py program and some more input images 
  to test the helmet detection module separately. 
* Major changes are made in the keras_yolo3 module including converting hyphen to underscore in the folder name 
  and adding __init__.py file to use it as a python module in the main GUI code. Significant changes made to 
  yolo.py (in detect_image() and detect_video() functions).


## Setup
Follow these instructions:  
I) Initial installation  

Clone this repo and then install the requirements:
`pip install -r requirements.txt`

If you are using Linux, you may need to install the following packages:
```console
$ sudo apt update 
$ sudo apt install libqt5x11extras5
$ sudo apt install libgl1-mesa-glx
```  
II) Downloading Large Files

1.  [yolov3.weights](https://drive.google.com/file/d/16XNa9Zt5GfgeCNW0fl8Hfx9ZaPQ2OEtt/view?usp=sharing). Put this file in the keras-yolo3/ subdirectory.  
2. [core](https://drive.google.com/file/d/17Uu7X9-MI0e0SetrV2ZReHOk1buWPfMH/view?usp=sharing). Put this file in the yolov3_Helmet_Detection/ subdirectory.  
3. [yolov3-obj_2400.weights](https://drive.google.com/file/d/16Pr_4FbOkoktDDE8rZpB8b2bh-GYouJl/view?usp=sharing). Put this file in the yolov3_Helmet_Detection/ subdirectory.
4. (todo- check if yolo.h5 is needed, demo video, detect_video independent, update screenshots)

III) Set up API key for ANPR  
Get your ANPR API key from [here](https://platerecognizer.com/?utm_source=github&utm_medium=website). Name this file as api_key.txt and place it in the current folder(yolov3_traffic_violation_tracking).


## Usage

To run the program, 
`python helmet_violation_monitoring_gui.py`
Above command launches the GUI from where you can run the program.  

![](screenshots/mainmenu.png?raw=true)  
Main menu of the GUI.

![](screenshots/database_entry.png?raw=true)
Database entry, based on which ANPR generates tickets/challans.

![](screenshots/helmet_detected.png?raw=true)  
Helmet Detected (green box)

![](screenshots/no_helmet.png?raw=true)  
Helmet Not Detected (popup indicates violation, shown in the ticket/challan below)

![](screenshots/challan.png?raw=true)  
Challan (ticket) generated by the program.