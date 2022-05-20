#!/bin/bash -i

pip install gdown
pip install -r requirements.txt
sudo apt update 
sudo apt install libqt5x11extras5
sudo apt install libgl1-mesa-glx

gdown '1ADn1oljvY_sjHrycz_xk7WWfVkXlErde'
gdown '18sgeHlupSZ2lgoqsVICegfaqqVyV89mT'
gdown '1ePhQZYf2y_4T7VlgwOoWgYNmk0_y_tro'

mv yolov3.weights keras-yolo3/yolov3.weights
mv core yolov3_Helmet_Detection/core
mv yolov3-obj_2400.weights yolov3_Helmet_Detection/yolov3-obj_2400.weights

cd keras_yolo3/
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
cd ..
