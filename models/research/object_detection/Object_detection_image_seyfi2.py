######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'a1a1.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')
def draw_bounding_box_on_image(image,
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
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.80)

# All the results have been drawn on image. Now display the image.

coordinates = vis_util.return_coordinates(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.80)
scale_percent = 20 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
ymin_1= int((boxes[0][0][0]*height*5))
xmin_1= int((boxes[0][0][1]*width*5))
ymax_1= int((boxes[0][0][2]*height*5))
xmax_1= int((boxes[0][0][3]*width*5))
ymin_2= int((boxes[0][1][0]*height*5))
xmin_2= int((boxes[0][1][1]*width*5))
ymax_2= int((boxes[0][1][2]*height*5))
xmax_2= int((boxes[0][1][3]*width*5))
xort_1=int((xmin_1+xmax_1)/2)
yort_1=int((ymin_1+ymax_1)/2)
xort_2=int((xmin_2+xmax_2)/2)
yort_2=int((ymin_2+ymax_2)/2)

ymin_3= int((boxes[0][2][0]*height*5))
xmin_3= int((boxes[0][2][1]*width*5))
ymax_3= int((boxes[0][2][2]*height*5))
xmax_3= int((boxes[0][2][3]*width*5))
xort_3=int((xmin_3+xmax_3)/2)
yort_3=int((ymin_3+ymax_3)/2)
cv2.line(image, (xmin_3, ymin_3), (xmax_3, ymax_3), (0, 255, 255), 2)

cv2.line(image, (xmin_1, ymin_1), (xmax_1, ymax_1), (0, 255, 255), 2)
cv2.line(image, (xmin_2, ymin_2), (xmax_2, ymax_2), (0, 255, 255), 2)

print(xmin_1)
print(xmin_2)
cv2.circle(image, (xort_1, yort_1),20, (0, 0, 255), 2)
cv2.circle(image, (xort_2, yort_2),20, (0, 0, 255), 2)
cv2.circle(image, (xort_3, yort_3),20, (0, 0, 255), 2)


resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#cv2.imshow('Object detector', image)

#print('Resized Dimensions : ',resized.shape)
#ymin = int((boxes[0][0][0]*height))
#xmin = int((boxes[0][0][1]*width))
#ymax = int((boxes[0][0][2]*height))
#xmax = int((boxes[0][0][3]*width))

#cv2.imshow("Resized image", resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imshow('Object detector', resized)

#textfile = open("json/"+filename_string+".json", "a")
#textfile.write(json.dumps(coordinates))
#textfile.write("\n")


# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
