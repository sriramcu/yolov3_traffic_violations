import sys
sys.path.append('../yolov3_Helmet_Detection/')
import Helmet_detection_YOLOV3 as hdy
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time 
import psutil

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

    
def detect_img(yolo,img,video_mode=False):

    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        
    else:
        r_image,all_boxes = yolo.detect_image(image)
        r_image.show()
        #print(video_mode)
        '''
        if video_mode:
            frame = np.array(r_image)
            # Create a 4D blob from a frame.
            inpWidth = 416       #Width of network's input image
            inpHeight = 416      #Height of network's input image
            blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net))

            # Remove the bounding boxes with low confidence
            boxes = hdy.postprocess(fn,frame, outs,outputFolder)
            #print("OK")
            yolo.close_session()
            
            return
        '''
        
        #below code written by sriramcu
        r_image.save('all_detections/'+img)
        plt.imshow(r_image)
        plt.show()
        
        bike_boxes = []
        rider_boxes = []
        #print(all_boxes) #label,left,top,right,bottom
        for box in all_boxes:
            label,left,top,right,bottom = box
            if 'motorbike' in label.lower():
                bike_boxes.append([left,top,right,bottom])
                
            elif 'person' in label.lower():
                rider_boxes.append([left,top,right,bottom])
                
                
            #im_crop = r_image.crop((left,top,right,bottom))
            #im_crop.show()

        #pairing then cropping then displaying each cropped image
        pairs = []
        
        for bb in bike_boxes:
            for rb in rider_boxes:
                l1 = Point(bb[0],bb[3]) #590,157
                r1 = Point(bb[2],bb[1]) #668, 270
                l2 = Point(rb[0],rb[3]) #631, 42
                r2 = Point(rb[2],rb[1]) #704, 236
                
                if(doOverlap(l1, r1, l2, r2)):                 
                    pairs.append([rb,bb])
                    
           
                    
                    
                    
                    
        print(pairs) 
        ctr=0
        time.sleep(2)
        for pair in pairs:
            ctr+=1
            image1 = Image.open(img)
            im_crop = image1.crop((max(pair[1][0]-30,0),max(pair[0][1]-30,0),pair[1][2]+30,pair[1][3]+30))
            im_crop.show()
            parts = img.rsplit('.',1)
            im_crop.save(os.path.join('cropped_images',parts[0]+str(ctr)+'.'+parts[1]))
            time.sleep(1)
            #im_crop.close()
            for proc in psutil.process_iter():
                if proc.name() == "display":
                    proc.kill()
        
        box_list = hdy.main('cropped_images','helmet_detections')
        time.sleep(3)
        for f in os.listdir('cropped_images'):
            if f not in os.listdir('helmet_detections') and f not in os.listdir('violations'):
      
                
                shutil.copy(os.path.join('cropped_images',f),os.path.join('violations',f))

        print(box_list)
            
        
            #save im_crop to a directory and run hdy.main(dir[0])
            

    yolo.close_session()



def detect_directory(path):
    pass
    
    
def detect_custom_video(video):
    pass
    
FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    for f in os.listdir('cropped_images'):
        os.remove(os.path.join('cropped_images',f))
    
    for f in os.listdir('all_detections'):
        os.remove(os.path.join('all_detections',f))
        
    #for f in os.listdir('violations'):
        #os.remove(os.path.join('violations',f))
        
    for f in os.listdir('helmet_detections'):
        os.remove(os.path.join('helmet_detections',f))
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    
    parser.add_argument(
        '--image', nargs='?', type = str,required=False,default="",
        help='Image detection mode, takes path of image(no defaults)' 
    )
    
    parser.add_argument(
        '--directory', type = str,
        help='Image detection mode of all images inside directory, default' + str(YOLO.get_defaults("directory"))
    )
    
    parser.add_argument(
        '--my_video', type = str,
        help='Detect violations in a video(no defaults) and put challans in a directory'
    )
    parser.add_argument(
        '--score', type = float,
        help='Confidence score(0 to 1), default ' + str(YOLO.get_defaults("score"))
    )
    
    '''parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    
    '''
    Command line positional arguments -- for video detection mode
    '''
    
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
  

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
       
        detect_img(YOLO(**vars(FLAGS)),FLAGS.image)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
