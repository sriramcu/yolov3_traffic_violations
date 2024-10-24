# Yolov3 Traffic Violations - Helmet Detection and ANPR

This application is an automated system of traffic monitoring to detect helmet violations. The system for 
detecting wrong side driving violations is implemented in YOLOv4 in 
[this repo](https://github.com/sriramcu/yolov4_wrong_side_driving_detection). The project 
automatically generates traffic violation tickets (or challans) based on registration details added to the 
database. 

## Acknowledgements

This project is based on three currently existing git repos:  
1. [BlcaKHat/yolov3-Helmet-Detection](https://github.com/BlcaKHat/yolov3-Helmet-Detection)
2. [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
3. [marcbelmont/deep-license-plate-recognition](https://github.com/marcbelmont/deep-license-plate-recognition)  

Each of the above repos have been modified to suit this application and can be found in their respective 
directories with their original README and LICENSE files. 

## Demo Video

Watch this video (todo) to see what this project does:

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
  person's bounding boxes with some buffer and stored inside `cropped_images` folder inside `keras_yolo3` folder.
* The ANPR program is run on these cropped images and moved into `violations` folder inside `keras_yolo3` folder, 
  where the new file names will be the computed license plate of the vehicle. 
* In case the license plate cannot be determined by the ANPR module, a suffix of "_unknown" is applied to 
  the filename inside the cropped images folder, so that ANPR can be skipped next time around.
* A challan is generated for each violation by referencing the `vehicles.db` sqlite3 database stored in 
  the root directory of the project using the license plate of the vehicle and is stored in challans 
  folder in the root directory of the project.
* Before running the helmet detection program, vehicle data is assumed to have been entered via the GUI, i.e. 
  vehicle license plates, name and address of the owner (3 columns).

### Changes made to the submodules referenced in the Acknowledgments Section

* The `deep_license_plate_recognition` module has minimal changes. 
* The `yolov3_Helmet_Detection` folder contains moderate changes, such as minor tweaks in the 
  `Helmet_detection_YOLOv3.py` program and some more input images to test the helmet detection module separately. 
* Major changes are made to the `keras_yolo3` module including converting hyphen to underscore in the folder name 
  and adding `__init__.py` file to use it as a python module in the main GUI code. Significant changes made to 
  `yolo.py` in `detect_image()` and `detect_video()` functions.


## Setup
Follow these instructions:  
**I) Initial installation**  

Clone this repo and then install the requirements:
`pip install -r requirements.txt`

If you are using Linux, you may need to install the following packages:
```console
$ sudo apt update 
$ sudo apt install libqt5x11extras5
$ sudo apt install libgl1-mesa-glx
```  
**II) Downloading Large Files**

1. [yolov3.weights](https://drive.google.com/file/d/1ncy-D_En32nMopdld9mXr972234LqMzd/view?usp=sharing). Put this 
file in the `keras-yolo3/` subdirectory.  
2. [core](https://drive.google.com/file/d/1T2sE0otIjKyagdP0dLCNtZ6u26fF6A_z/view?usp=sharing). Put this file in 
   the `yolov3_Helmet_Detection/` subdirectory.  
3. [yolov3-obj_2400.weights](https://drive.google.com/file/d/1bj8dTxl1anTF0U1Tq3awx7bnWOGUWEJa/view?usp=sharing)
   . Put this file in the `yolov3_Helmet_Detection/` subdirectory.
4. [yolo.h5](https://drive.google.com/file/d/1qgoUvCT2ajo4lkFknRTYRoKrthXoUvNe/view?usp=sharing). Put this file 
   in the `keras-yolo3/model_data/` subdirectory.

**III) Set up API key for ANPR**  
Get your ANPR API key from [here](https://platerecognizer.com/?utm_source=github&utm_medium=website). Name this 
file as `api_key.txt` and place it in the root directory of the project.


## Usage

(All the below commands are run in the root directory of the project as the current working directory)

**I) To run the main GUI program,**  

`python helmet_violation_monitoring_gui.py`

Refer to the demo video and the high level overview to understand the features of the GUI - run helmet 
detection (by selecting input video file in the file picker and timestamped output video file's location using 
the folder picker), add vehicle entry into database, read database and generate challans.  

The constants.py file in `keras_yolo3/` submodule contains `COMPUTATION_FPS`, which is an assumed value for the 
speed at which your system processes a given video. Before trying real time applications, see if this value is 
correct by seeing the output printed by the above program, which mentions this assumed computation FPS, the 
actual computation FPS measured on a test input video, and the FPS of the input video file. Then, with minimal 
changes to the `yolo.py` program, you can try real time applications.  

The lower your computation fps, the shorter your output video will be since it saves fewer frames into the 
output video while yielding faster processing.

---

II) To run the **helmet violations tracking module separately** on the command line,  
`python keras_yolo3/yolo_video.py keras_yolo3/input_videos/demo_input.mp4 keras_yolo3/output_videos/demo_output_cmd.mp4`

III) To run just the **helmet detection on a batch of images without any overlap logic**,  
`python yolov3_Helmet_Detection/Helmet_detection_YOLOV3.py`

By default input images are stored in `yolov3_Helmet_Detection/images` folder and output images are stored in 
`yolov3_Helmet_Detection/test_out` folder.

IV) To run just the **ANPR on a batch of images**, stored in `keras_yolo3/cropped_images` folder,  
`python run_lpr.py`.   

Some additional images are stored in `keras_yolo3/input_frames` folder, which you can 
manually move to the `keras_yolo3/cropped_images` folder before running the program. As mentioned in the high 
level overview section, the violations will be stored in `keras_yolo3/violations` folder. Challans can then be 
generating using the "Generate Challans" button in the main GUI without having to run the helmet 
violations tracking via the GUI.

---

## Output

![](screenshots/challan.png?raw=true)  
Challan (ticket) generated by the program.

## Contributions / Future Scope

Open source contributions or PRs for the project are welcome. Two features that can potentially be implemented, 
in addition to GUI enhancements, are:
1. Easier use of real time applications configurable on the GUI or via the command line, such as via a webcam 
   or a Rpi device.
2. Automatic computation FPS without assumptions as mentioned in the usage section.
