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
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from imutils.video import VideoStream
from utils import label_map_util
from utils import visualization_utils as vis_util


#Telegram options
chatid = 916115485
bot = telegram.Bot(token='1011036996:AAGfiM6yIOfjJFggRmImKwyoAoDMfOpc5EU')
start = time.time()

# Define the video stream
#cap=VideoStream('http://192.168.43.238:8080/video')
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Download Model



# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            #image_np = cap.read()
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
            isim=[]
            for index, value in enumerate(classes[0]):
                object_dict = {}
                if scores[0, index] > 0.8:
                    
                    isim_dict = {}
                    object_dict[(category_index.get(value)).get('name').encode('utf8')] = scores[0, index]
                    #object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                        #scores[0, index]
                    objects.append(object_dict)
                    isim_dict = (category_index.get(value)).get('name').encode('utf8')
                    isim.append(isim_dict)


                    sayi=int(len(isim))
                    #print(type(sayi))
                    #print(sayi)

                    axmax=[None] * len(isim)
                    aymax=[None] * len(isim)
                    axmin=[None] * len(isim)
                    aymin=[None] * len(isim)
                    axort=[None] * len(isim)
                    ayort=[None] * len(isim)
                    b=0
                    a=100


                    if len(isim)==3:
                        name1=str(isim[0])
                        name2=str(isim[1])
                        name3 = str(isim[2])

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
                            #print(isim[b], axort[b], ayort[b])
                            b=b+1
                        if (name1==name3=="b'insan'" and name2=="b'canta'"):
                            cv2.circle(image_np, (axort[1], ayort[1]), 10, (0, 0, 255), 5)

                            if ((axort[1] - a < axort[0] < axort[1] + a) and (ayort[1]-a<ayort[0] < ayort[1]+a))and ((axort[1] - a < axort[2] < axort[1] + a) and (ayort[1]-a<ayort[2]<ayort[1]+a)):
                                print('\nCanta 2 de')
                                print(isim[0], axort[0], ayort[0])
                                print(isim[1], axort[1], ayort[1])
                                print(isim[2], axort[2], ayort[2])
                                print("cok yakin")

                                img_name = "Hedef.jpeg"
                                cv2.imwrite(img_name, image_np)
                                start = time.time()
                                bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                bot.send_photo(chat_id=chatid, photo=open('Hedef.jpeg', 'rb'), timeout=100)


                        elif (name2==name3=="b'insan'" and name1=="b'canta'"):
                            cv2.circle(image_np, (axort[0], ayort[0]), 10, (0, 0, 255), 5)

                            if ((axort[0]-a<axort[1] < axort[0]+a) and (ayort[0]-a < ayort[1] < ayort[0]+a)) and ((axort[0]-a<axort[2] < axort[0]+a) and (ayort[0]-a < ayort[2] < ayort[0]+a)):
                                print('\nCanta 1 de')
                                print(isim[0], axort[0], ayort[0])
                                print(isim[1], axort[1], ayort[1])
                                print(isim[2], axort[2], ayort[2])
                                print("cok yakin")
                                img_name = "Hedef.jpeg"
                                cv2.imwrite(img_name, image_np)
                                start = time.time()
                                bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                bot.send_photo(chat_id=chatid, photo=open('Hedef.jpeg', 'rb'), timeout=100)

                        elif (name2 == name1 == "b'insan'" and name3 == "b'canta'"):

                            cv2.circle(image_np, (axort[2], ayort[2]), 10, (0, 0, 255), 10)

                            if ((axort[2] - a < axort[1] < axort[2] + a) and (
                                    ayort[2] - a < ayort[1] < ayort[2] + a)) and (
                                    (axort[2] - a < axort[0] < axort[2] + a) and (
                                    ayort[2] - a < ayort[0] < ayort[2] + a)):

                                print('\nCanta 3 de')
                                print(isim[0], axort[0], ayort[0])
                                print(isim[1], axort[1], ayort[1])
                                print(isim[2], axort[2], ayort[2])
                                print("cok yakin")
                                img_name = "Hedef.jpeg"
                                cv2.imwrite(img_name, image_np)
                                start = time.time()
                                bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                bot.send_photo(chat_id=chatid, photo=open('Hedef.jpeg', 'rb'), timeout=100)






                                #name = str(isim)
            #print(objects)
            #print(bool(objects))
            #print(name)
            #if bool(isim)==True:
            #print (isim[0])
            resized = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('object detection', image_np)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
