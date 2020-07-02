######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

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
import time
import argparse
import posenet



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites

from utils import label_map_util
from utils import visualization_utils as vis_util


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=500)
parser.add_argument('--cam_height', type=int, default=500)
parser.add_argument('--scale_factor', type=float, default=1) #0.7125
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()



# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
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

    while(True):
        fps = time.time()
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        
        input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride) 
        
        
        
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess2.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )           
        
        
        
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.15)
    
        keypoint_coords *= output_scale
        frame_expanded = np.expand_dims(display_image, axis=0)
    
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
    
        
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            display_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)
        
        b= []
            
            
        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image, b = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords, b,
            min_pose_score=0.15, min_part_score=0.1)
        
        
        nose = [None]*10
        left_eye = [None]*10
        right_eye = [None]*10
        left_wrist = [None]*10
        right_wrist = [None]*10
        face = [None]*10
        fark = 0
        
        
        for i in range(len(b)):
            if b[i] == []:
                continue
            for m in range(len(b[i])):
                
                if b[i][m][0] == 0:
                    nose[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                    
                if b[i][m][0] == 2:
                    left_eye[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                
                if b[i][m][0] == 1:
                    right_eye[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                    
                if b[i][m][0] == 9:
                    left_wrist[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                
                if b[i][m][0] == 10:
                    right_wrist[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                
        
        for i in range(len(nose)):   
            if nose[i] != None and left_eye[i] !=  None  and right_eye[i] !=  None:
                fark = abs(left_eye[i][0]-right_eye[i][0])
                start_point = (int(left_eye[i][0] - fark), nose[i][1] + int(fark*1.5))
                end_point= (int(right_eye[i][0] + fark), nose[i][1] - int(fark*2.5))                
                overlay_image = cv2.rectangle(overlay_image, start_point, end_point, color=(0,150,0), thickness=2)  
                face[i] = [start_point, end_point]                
        
        
              
        pfps = time.time() - fps
        pfps = str(int(1/pfps))        
        print(pfps)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', overlay_image)
    
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break       
        
    # Clean up
    cap.release()
    cv2.destroyAllWindows()


        
        
                 
           
            
