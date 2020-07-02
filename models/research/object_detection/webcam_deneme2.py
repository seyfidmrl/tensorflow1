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
import dlib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import imutils
sys.path.append("..")
from object_detection.utils import ops as utils_ops
import argparse
from utils import label_map_util
from utils import visualization_utils as vis_util

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())
#Telegram options

start = time.time()

# Define the video stream
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

writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
startX=0
startY=0
endX=0
endY=0
				
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            frame = cap.read()
            frame = frame[1]

            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video


            # resize the frame to have a maximum width of 500 pixels (the
            # less data we have, the faster we can process it), then convert
            # the frame from BGR to RGB for dlib
            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize
            # the writer


            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Bekleniyor"
            rects = []

            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker

            if totalFrames % args["skip_frames"] == 0:
                # set the status and initialize our new set of object trackers
                status = "Algilaniyor"
                trackers = []

                # convert the frame to a blob and pass the blob through the
                # network and obtain the detections
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)


                # loop over the detections

                # loop over the detections

            # Read frame from camera
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
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Display output
            objects = []
            isim=[]
            
            for index, value in enumerate(classes[0]):
                object_dict = {}
                if scores[0, index] > 0.5:
                    
                    isim_dict = {}
                    #object_dict[(category_index.get(value)).get('name').encode('utf8')] = scores[0, index]
                    #object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                        #scores[0, index]
                    #objects.append(object_dict)
                    isim_dict = (category_index.get(value)).get('name').encode('utf8')
                    isim.append(isim_dict)
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

                    for tracker in trackers:
                        # set the status of our system to be 'tracking' rather
                        # than 'waiting' or 'detecting'
                        status = "izleniyor"

                        # update the tracker and grab the updated position
                        tracker.update(rgb)
                        pos = tracker.get_position()

                        # unpack the position object
                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())

                        # add the bounding box coordinates to the rectangles list

                        print(endX)
                        print(endY)

                # draw a horizontal line in the center of the frame -- once an
                # object crosses this line we will determine whether they were
                # moving 'up' or 'down'
                    #cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
                # cv2.line(frame, (0, H//2), (W, H//2), (0, 255, 255), 2)

                # use the centroid tracker to associate the (1) old object
                # centroids with (2) the newly computed object centroids

                        
		    
                    if len(isim)==3:
                        name1=str(isim[0])
                        name2=str(isim[1])
                        name3 = str(isim[2])
                        if (name1=="b'insan'" and name2=="b'insan'" and name3=="b'canta'") or (name1=="b'insan'" and name2=="b'canta'" and name3=="b'insan'") or (name1=="b'canta'" and name2=="b'insan'" and name3=="b'insan'") :
                    
                            print ('TEHLİKELİ DURUM OLABILIR')
                        


            #print(endX)
            #print(endY)
            #name = str(isim)
            #print(objects)
            #print(bool(objects))
            #print(name)
            #if bool(isim)==True:
            #print (isim[0])

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))


            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
