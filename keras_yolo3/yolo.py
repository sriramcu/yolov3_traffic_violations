# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from keras.models import load_model

import yolov3_Helmet_Detection.Helmet_detection_YOLOV3 as hdy
from keras_yolo3.constants import COMPUTATION_FPS
from keras_yolo3.rectangle_operations import do_overlap, Point
from keras_yolo3.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from keras_yolo3.yolo3.utils import letterbox_image

overlapping_frames_counter = 0  # used as filename of cropped violation detected, keeps count b/w many function calls
CROPPED_IMAGES_DIRECTORY = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'cropped_images')


class YOLO(object):
    _defaults = {
        "model_path": os.path.join(os.path.dirname(__file__), 'model_data', 'yolo.h5'),
        "anchors_path": os.path.join(os.path.dirname(__file__), 'model_data', 'yolo_anchors.txt'),
        "classes_path": os.path.join(os.path.dirname(__file__), 'model_data', 'coco_classes.txt'),
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
        "directory": './images_test'
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
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
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except Exception:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
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
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        img_arr = np.array(image)
        img_arr1 = np.array(image)
        input_width = 416  # Width of network's input image
        input_height = 416  # Height of network's input image
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(  # type: ignore
            img_arr, 1 / 255, (input_width, input_height), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        hdy.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = hdy.net.forward(hdy.getOutputsNames(hdy.net))

        # Remove the bounding boxes with low confidence
        helmet_boxes, score_boxes = hdy.postprocess('', img_arr, outs, '', True)

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        for ind, hbx in enumerate(helmet_boxes):
            hbx = [hbx[1], hbx[0], hbx[1] + hbx[3], hbx[0] + hbx[2]]
            a = np.array(hbx)
            a = np.reshape(a, (1, a.size))
            out_boxes = np.concatenate((out_boxes, a))
            out_classes = np.concatenate((out_classes, np.array([80])))
            out_scores = np.concatenate((out_scores, np.array([score_boxes[ind]])))
            # detected helmets' boxes, confidence scores should be displayed regardless of threshold

        font = ImageFont.truetype(
            font=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'font', 'FiraMono-Medium.otf'),
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        all_boxes = []

        for i, c in reversed(list(enumerate(out_classes))):
            if c == 80:
                predicted_class = 'Helmet'
                c = 35  # for colour purpose
            else:
                predicted_class = self.class_names[c]

            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textlength(label, font)  # type: ignore

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            all_boxes.append([label, left, top, right, bottom])

            if top - label_size >= 0:
                text_origin = np.array([left, top - label_size])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j],
                    outline=self.colors[c])  # type: ignore
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])  # type: ignore
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # type: ignore
            del draw

        bike_boxes = []
        rider_boxes = []
        helmet_boxes = []
        for box in all_boxes:
            label, left, top, right, bottom = box
            if 'motorbike' in label.lower():
                bike_boxes.append([left, top, right, bottom])

            elif 'person' in label.lower():
                rider_boxes.append([left, top, right, bottom])

            elif 'helmet' in label.lower():
                helmet_boxes.append([left, top, right, bottom])

        for bike_box in bike_boxes:
            helmet_detected_for_any_rider = False
            for rider_box in rider_boxes:
                l1 = Point(bike_box[0], bike_box[3])  # 590,157
                r1 = Point(bike_box[2], bike_box[1])  # 668, 270
                l2 = Point(rider_box[0], rider_box[3])  # 631, 42
                r2 = Point(rider_box[2], rider_box[1])  # 704, 236

                if do_overlap(l1, r1, l2, r2):  # bike, rider

                    for helmet_box in helmet_boxes:
                        l1 = Point(helmet_box[0], helmet_box[3])  # 590,157
                        r1 = Point(helmet_box[2], helmet_box[1])  # 668, 270
                        l2 = Point(rider_box[0], rider_box[3])  # 631, 42
                        r2 = Point(rider_box[2], rider_box[1])  # 704, 236

                        if do_overlap(l1, r1, l2, r2):  # helmet, rider
                            helmet_detected_for_any_rider = True
                            break

            if not helmet_detected_for_any_rider:
                for rider_box in rider_boxes:
                    l1 = Point(bike_box[0], bike_box[3])  # 590,157
                    r1 = Point(bike_box[2], bike_box[1])  # 668, 270
                    l2 = Point(rider_box[0], rider_box[3])  # 631, 42
                    r2 = Point(rider_box[2], rider_box[1])  # 704, 236

                    if do_overlap(l1, r1, l2, r2):
                        # crop
                        global overlapping_frames_counter
                        overlapping_frames_counter += 1
                        pair = [rider_box, bike_box]
                        I = cv2.cvtColor(img_arr1, cv2.COLOR_BGR2RGB)
                        image2 = Image.fromarray(I, mode='RGB')
                        im_crop = image2.crop(
                            (max(pair[1][0] - 30, 0), max(pair[0][1] - 30, 0), pair[1][2] + 30, pair[1][3] + 30))
                        # crops the combined image of the violating rider and bike out of the main frame
                        im_crop.save(os.path.join(CROPPED_IMAGES_DIRECTORY, str(overlapping_frames_counter) + '.jpg'))
                        break

        return image, all_boxes

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_four_cc = int(vid.get(cv2.CAP_PROP_FOURCC))
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    is_output_stored = True if output_path != "" else False
    vid_output_stream = None
    if is_output_stored:
        print("!!! TYPE:", type(output_path), type(video_four_cc), type(video_fps), type(video_size))
        vid_output_stream = cv2.VideoWriter(output_path, 0x7634706d, video_fps, video_size)

    start_time = timer()
    frame_counter = 0  # used to count number of frames for which detection has been skipped
    computation_fps = COMPUTATION_FPS
    time_per_frame = 1 / video_fps
    time_per_computation_frame = 1 / computation_fps
    next_computation_time = 0
    while True:
        frame_counter += 1
        ret, frame = vid.read()
        if not ret:
            break

        current_video_time = frame_counter * time_per_frame

        if current_video_time < next_computation_time:
            print(f"Skipping frame {frame_counter}; current_video_time = {current_video_time:.2f}")
            continue
        try:
            image = Image.fromarray(frame)
        except AttributeError as e:
            print(e)
            yolo.close_session()
            return

        image, _ = yolo.detect_image(image)
        displayed_frame = np.array(image)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", displayed_frame)
        if is_output_stored:
            vid_output_stream.write(displayed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_counter >= total_frames:
            break
        next_computation_time += time_per_computation_frame

    actual_computation_fps = frame_counter / (timer() - start_time)
    actual_computation_fps = round(actual_computation_fps, 2)
    print(f"Video processing complete. Assumed computation FPS = {computation_fps}, "
          f"actual computation FPS = {actual_computation_fps}, source video FPS = {video_fps}")
    yolo.close_session()
