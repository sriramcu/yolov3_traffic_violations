# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
from time import sleep
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../yolov3_Helmet_Detection/')
import Helmet_detection_YOLOV3 as hdy
import psutil
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


ctr1 = 0
class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
  
# Returns true if two rectangles(l1, r1)  
# and (l2, r2) overlap 
def doOverlap(l1, r1, l2, r2): 
      
    # If one rectangle is on left side of other 
    if(l1.x >= r2.x or l2.x >= r1.x): 
        
        return False
  
    # If one rectangle is above other 
    if(l1.y <= r2.y or l2.y <= r1.y): 
  
        return False
  
    return True
    
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
        "directory" : './images_test'
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            # self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
            print("Multiple GPUs are not supported at this time. Please run the code using only 1 GPU.")
            sys.exit(-1)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):   #uncomment lines 110,126,127 if helmet detection not needed
        start = timer()
        #'''
        
        img_arr = np.array(image)
        img_arr1 = np.array(image)
        frame_count =0
        inpWidth = 416       #Width of network's input image
        inpHeight = 416      #Height of network's input image
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(img_arr, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        hdy.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = hdy.net.forward(hdy.getOutputsNames(hdy.net))

        # Remove the bounding boxes with low confidence
        helmet_boxes,score_boxes = hdy.postprocess('',img_arr, outs,'',True)
        #'''
        #helmet_boxes = []
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        first = 0
        for ind,hbx in enumerate(helmet_boxes):
            if first == 0:
                #print(helmet_boxes)
                first = 1
            hbx = [hbx[1],hbx[0],hbx[1]+hbx[3],hbx[0]+hbx[2]]
            a = np.array(hbx)
            a = np.reshape(a,(1, a.size))
            out_boxes = np.concatenate((out_boxes,a))
            out_classes = np.concatenate((out_classes,np.array([80])))
            out_scores = np.concatenate((out_scores,np.array([score_boxes[ind]])))
            
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        all_boxes = []
        for i, c in reversed(list(enumerate(out_classes))):
            if c != 80:
                predicted_class = self.class_names[c]
                
                
            else:
                predicted_class = 'Helmet'
                c = 35   #for colour purpose
                
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            all_boxes.append([label,left,top,right,bottom])


            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw


        
        end = timer()
        #print(end - start)
        bike_boxes = []
        rider_boxes = []
        helmet_boxes = []
        #print(all_boxes) #label,left,top,right,bottom
        for box in all_boxes:
            label,left,top,right,bottom = box
            if 'motorbike' in label.lower():
                bike_boxes.append([left,top,right,bottom])
                
            elif 'person' in label.lower():
                rider_boxes.append([left,top,right,bottom])
                
            elif 'helmet' in label.lower():
                helmet_boxes.append([left,top,right,bottom])
                
        for bb in bike_boxes:
            flag = 0
            for rb in rider_boxes:
                l1 = Point(bb[0],bb[3]) #590,157
                r1 = Point(bb[2],bb[1]) #668, 270
                l2 = Point(rb[0],rb[3]) #631, 42
                r2 = Point(rb[2],rb[1]) #704, 236
                
                if(doOverlap(l1, r1, l2, r2)): 
                     
                
                    for hb in helmet_boxes:
                        l1 = Point(hb[0],hb[3]) #590,157
                        r1 = Point(hb[2],hb[1]) #668, 270
                        l2 = Point(rb[0],rb[3]) #631, 42
                        r2 = Point(rb[2],rb[1]) #704, 236   
                        
                        if(doOverlap(l1, r1, l2, r2)):
                            flag = 1
                            break
                            
            if not flag:
                for rb in rider_boxes:
                    l1 = Point(bb[0],bb[3]) #590,157
                    r1 = Point(bb[2],bb[1]) #668, 270
                    l2 = Point(rb[0],rb[3]) #631, 42
                    r2 = Point(rb[2],rb[1]) #704, 236
                    
                    if(doOverlap(l1, r1, l2, r2)): 
                        #crop
                        global ctr1
                        ctr1+=1
                        pair = [rb,bb]
                        I = cv2.cvtColor(img_arr1, cv2.COLOR_BGR2RGB)
                        image2 = Image.fromarray(I,mode='RGB')
                        im_crop = image2.crop((max(pair[1][0]-30,0),max(pair[0][1]-30,0),pair[1][2]+30,pair[1][3]+30))
                        # Comment out below line to disable challan images popping up during execution
                        # todo control this using argparse
                        im_crop.show()
                        
                        #parts = img.rsplit('.',1)
                        #im_crop.save(os.path.join('cropped_images',parts[0]+str(ctr)+'.'+parts[1]))
                        im_crop.save(os.path.join('cropped_images',str(ctr1)+'.jpg'))
                        sleep(1)
                        #im_crop.close()
                        for proc in psutil.process_iter():
                            if proc.name() == "display":
                                proc.kill()
                        break
                
            
        return image,all_boxes

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, 0x7634706d, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    skip_counter = -1
    
    fps_list = []
    common_fps = video_fps #initially asssume output is real time
    while True:
        
        skip_counter+=1
        return_value, frame = vid.read()

        if skip_counter%(int(video_fps/common_fps)) != 0:
            print("Skipping frame;skip_counter = "+str(skip_counter))
            #out.write(frame)   #unlabelled frames for output video to be of same length
            continue
        try:
            image = Image.fromarray(frame)
        except AttributeError:
            if isOutput:
                print("Writing to output file...")
                out.write(result)
     
            yolo.close_session()
            return
            
        image,discard = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            fps_list.append(curr_fps)
            curr_fps = 0

        print(fps)
        
        if len(fps_list) >= 7:
            common_fps = max(set(fps_list), key=fps_list.count) #most common element of list
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        #plt.imshow(result) 
            
        #plt.show()
        #print(type(result))
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

