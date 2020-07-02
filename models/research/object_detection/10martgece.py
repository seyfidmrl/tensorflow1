import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import time
import telegram
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import imutils
import argparse
import posenet


sys.path.append("..")
from object_detection.utils import ops as utils_ops
from imutils.video import VideoStream
from utils import label_map_util
from utils import visualization_utils as vis_util
chatid = 916115485
bot = telegram.Bot(token='1011036996:AAGfiM6yIOfjJFggRmImKwyoAoDMfOpc5EU')
start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=500)
parser.add_argument('--cam_height', type=int, default=500)
parser.add_argument('--scale_factor', type=float, default=1) #0.7125
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
NUM_CLASSES = 2
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
         (im_height, im_width, 3)).astype(np.uint8)

with tf.Session() as sess2:
    model_cfg, model_outputs = posenet.load_model(args.model, sess2)
    output_stride = model_cfg['output_stride']
    
    if args.file is not None:
        cap = cv2.VideoCapture(args.file)
    else:
        cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width) #640-480
    cap.set(4, args.cam_height)
    height,width = 480,640


j = 0
k = 0


mesafe = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
bag = 0
owner = 0
suspect = 0
distance=100

# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            # image_np = cap.read()
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})


            def draw_bounding_box_on_image(image_np,
                                           ymin,
                                           xmin,
                                           ymax,
                                           xmax,
                                           color='red',
                                           thickness=4,
                                           display_str_list=(),
                                           use_normalized_coordinates=True):
                """Adds a bounding box to an image.
                Bounding box coordinates can be specified in either absolute (pixel) or
                normalized coordinates by setting the use_normalized_coordinates argument.
                Each string in display_str_list is displayed on a separate line above the
                bounding box in black text on a rectangle filled with the input 'color'.
                If the top of the bounding box extends to the edge of the image, the strings
                are displayed below the bounding box.
                Args:
                  image: a PIL.Image object.
                  ymin: ymin of bounding box.
                  xmin: xmin of bounding box.
                  ymax: ymax of bounding box.
                  xmax: xmax of bounding box.
                  color: color to draw bounding box. Default is red.
                  thickness: line thickness. Default value is 4.
                  display_str_list: list of strings to display in box
                                    (each to be shown on its own line).
                  use_normalized_coordinates: If True (default), treat coordinates
                    ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
                    coordinates as absolute.
                """
                draw = ImageDraw.Draw(image_np)
                im_width, im_height = image_np.size
                if use_normalized_coordinates:
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                  ymin * im_height, ymax * im_height)


            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.80)
            coordinates = vis_util.return_coordinates(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.80)
            scale_percent = 20  # percent of original size
            width = int(image_np.shape[1] * scale_percent / 100)
            height = int(image_np.shape[0] * scale_percent / 100)
            dim = (width, height)

            # Display output
            objects = []
            isim = []

            for index, value in enumerate(classes[0]):
                object_dict = {}
                if scores[0, index] > 0.8:

                    isim_dict = {}
                    object_dict[(category_index.get(value)).get('name').encode('utf8')] = scores[0, index]
                    # object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                    # scores[0, index]
                    objects.append(object_dict)
                    isim_dict = (category_index.get(value)).get('name').encode('utf8')
                    isim.append(isim_dict)

                    sayi = int(len(isim))
                    # print(len(isim))
                    # print(type(sayi))
                    # print(sayi)

                    axmax = [None] * len(isim)
                    aymax = [None] * len(isim)
                    axmin = [None] * len(isim)
                    aymin = [None] * len(isim)
                    axort = [None] * len(isim)
                    ayort = [None] * len(isim)
                    b = 0
                    a = 100

                    if (len(isim) == 3 and len(isim) != 4):
                        obje0 = str(isim[0])
                        obje1 = str(isim[1])
                        obje2 = str(isim[2])

                        for i in range(3):
                            # name1=str(isim[0])
                            # name2=str(isim[1])
                            # name3 = str(isim[2])

                            aymin[b] = int((boxes[0][b][0] * height * 5))
                            axmin[b] = int((boxes[0][b][1] * width * 5))
                            aymax[b] = int((boxes[0][b][2] * height * 5))
                            axmax[b] = int((boxes[0][b][3] * width * 5))
                            axort[b] = int((axmin[b] + axmax[b]) / 2)
                            ayort[b] = int((aymin[b] + aymax[b]) / 2)

                            b = b + 1

                        farkx10 = abs(axort[1] - axort[0])
                        farkx12 = abs(axort[1] - axort[2])
                        farkx20 = abs(axort[2] - axort[0])

                        farky10 = abs(ayort[1] - ayort[0])
                        farky12 = abs(ayort[1] - ayort[2])
                        farky20 = abs(ayort[2] - ayort[0])

                        mesafe[1][0] = mesafe[0][1] = int(pow((farkx10 * farkx10 + farky10 * farky10), 0.5))
                        mesafe[2][0] = mesafe[0][2] = int(pow((farkx20 * farkx20 + farky20 * farky20), 0.5))
                        mesafe[1][2] = mesafe[1][2] = int(pow((farkx12 * farkx12 + farky12 * farky12), 0.5))
            resized = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('object detection', image_np)
