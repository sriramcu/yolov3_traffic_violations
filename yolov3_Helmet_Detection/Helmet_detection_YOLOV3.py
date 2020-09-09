from time import sleep
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
frame_count = 0             # used in mainloop  where we're extracting images., and then to drawPred( called by post process)
frame_count_out=0           # used in post process loop, to get the no of specified class value.
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


# Load names of classes
#classesFile = "../yolov3-Helmet-Detection/obj.names"
classesFile = os.path.join(os.path.dirname(__file__),'obj.names') #obj.names file must be in the same directory as this module
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = os.path.join(os.path.dirname(__file__),'yolov3-obj.cfg')
modelWeights = os.path.join(os.path.dirname(__file__),'yolov3-obj_2400.weights')

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box


def drawPred(frame,classId, conf, left, top, right, bottom):

    global frame_count
# Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    #print(label)            #testing
    #print(labelSize)        #testing
    #print(baseLine)         #testing

    label_name,label_conf = label.split(':')    #spliting into class & confidance. will compare it with person.
    if label_name == 'Helmet':
                                            #will try to print of label have people.. or can put a counter to find the no of people occurance.
                                        #will try if it satisfy the condition otherwise, we won't print the boxes or leave it.
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        frame_count+=1


    if(frame_count> 0):
        return frame_count,label




# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(fn,frame, outs,saved_folder = 'test_out',only_boxes=False):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out=0
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []               #have to fins which class have hieghest confidence........=====>>><<<<=======
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                #print(classIds)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person=0 # for counting the classes in this loop.
    my_scores = []
    helmet_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        helmet_boxes.append(box)
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
               #this function in  loop is calling drawPred so, try pushing one test counter in parameter , so it can calculate it.
        frame_count_out,lbl = drawPred(frame,classIds[i], confidences[i], left, top, left + width, top + height)
    
        my_scores.append(confidences[i])
         #increase test counter till the loop end then print...

        #checking class, if it is a person or not

        my_class='Helmet'                   #======================================== mycode .....
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1


    
    if count_person >= 1:
        if only_boxes:
            '''
            full_img = frame[:,:,[2,1,0]]  #BGR to RGB
            print("Reached")
            plt.imshow(full_img) 
            
            plt.show()
            #(Image.fromarray(frame, 'RGB')).show()
            #sleep(10)
            '''

            return helmet_boxes,my_scores
        
        path = saved_folder
        frame_name=os.path.basename(fn)             # trimm the path and give file name.
        parts = frame_name.rsplit('.',1)

        
        
        
        
        if not cv.imwrite(os.path.join(str(path),parts[0]+'.'+parts[1]), frame):     # writing to folder.
            raise FileNotFoundError
        
        print("Not ok")
        #cv.imshow('img',frame)
        full_img = frame[:,:,[2,1,0]]  #BGR to RGB
       
        plt.imshow(full_img) 

        plt.show()
        cv.waitKey(800)

    return helmet_boxes,my_scores
        
    #return boxes,my_scores #2d array of boxes associated with a file


    #cv.imwrite(frame_name, frame)
                                               #======================================mycode.........


# Process inputs
def main(folderName='images',outputFolder ='test_out'):
    winName = 'Deep learning object detection in OpenCV'


    #cv.namedWindow(winName, cv.WINDOW_NORMAL)


    box_list = []
    full_imgs = []
    t_vals = []
    for fn in glob(folderName+'/*'):
        if 'txt' in fn:
            continue
        frame = cv.imread(fn)
        frame_count =0

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        boxes = postprocess(fn,frame, outs,outputFolder)
       
        box_list.append(boxes)

        
     

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()

        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    
    return box_list

        

if __name__ == '__main__':
    print(main())


