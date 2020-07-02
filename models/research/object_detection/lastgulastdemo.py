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
from datetime import datetime
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

dosya=open('aar.txt','r+',encoding="utf-8")

dizi=dosya.readlines()
# print(dizi[0])
# print(dizi[1])
# print(dizi[2])
#telid=(dosya.readline())
telid=dizi[0]
#cameraid=(dosya.readline())
cameraid=dizi[1]
#Maskt=(dosya.readline())
Maskt=dizi[2]
dosya.close()
#Telegram options
#chatid = 916115485
clean = telid.replace(r"\n", " ")
chatid =int (clean)
print(chatid)
#print(type(chatid))
#print(cameraid)

if(cameraid == "1\n"):
    kamk = 0
    print(kamk)
elif(cameraid=='2\n'):
     kamk="http://192.168.1.20:8080/video"
     print(kamk)
elif(cameraid=='3\n'):
     kamk="http://192.168.1.36:8080/video"
     print(kamk)
if(Maskt=='On\n'):
     print('Mask Detection Start')
elif(Maskt=='Off\n'):
     print('Mask Detection Stop')


bot = telegram.Bot(token='1011036996:AAGfiM6yIOfjJFggRmImKwyoAoDMfOpc5EU')
start = time.time()

# Define the video stream
#cap=VideoStream('http://192.168.43.238:8080/video')
cap = cv2.VideoCapture(kamk)  # Change only if you have more than one webcams

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

sonuc_1 = [None] * 1500
sonuc_2 = [None] * 1500
sonuc_3 = [None] * 1500
sonuc_4 = [None] * 1500
sonuc_5 = [None] * 1500
sonuc_6 = [None] * 1500
sonuc_7 = [None] * 1500
sonuc_8 = [None] * 1500
sonuc_9 = [None] * 1500
j=0
k=0
l=0
m=0
n=0
o=0
p=0
r=0
s=0
vs = 0;


# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            #image_np = cap.read()
            ret, image_np = cap.read()
            resim=image_np
            resim_name="cap.jpg"
            cv2.imwrite(resim_name,resim)
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
                    #print(len(isim))
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

                    


                    if (len(isim)==3 and len(isim)!=4):
                        obje0=str(isim[0])
                        obje1=str(isim[1])
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

                        mesafe10 = pow((farkx10 * farkx10 + farky10 * farky10), 0.5)
                        mesafe20 = pow((farkx20 * farkx20 + farky20 * farky20), 0.5)
                        mesafe12 = pow((farkx12 * farkx12 + farky12 * farky12), 0.5)
                        
                        cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                        img_name = "aa.jpg"
                        cv2.imwrite(img_name, cropped)
                        print("sdsdsds")

                            #print(isim[b], axort[b], ayort[b])

                        if ((obje0==obje2=="b'person'" and obje1=="b'bag'") and (mesafe10 > mesafe12)):
                            cv2.circle(image_np, (axort[1], ayort[1]), 10, (0, 0, 255), 5)
                            sonuc_1[j] = int(mesafe10 - mesafe12)
                            j = j + 1
                            if (j > 3):
                                if (((sonuc_1[j - 1] > 0 and sonuc_1[j - 2] < 0)) or ((sonuc_1[j - 1] < 0 and sonuc_1[j - 2] > 0))):
                                    
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name = "deneme/face/"+ dt_string + ".jpg"
                                    cv2.imwrite(img_name, image_np)
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    resim=cv2.imread("cap.jpg")
                                    cropped=resim[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    resim=cv2.imread("cap.jpg")
                                    cropped=resim[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    
                                    #cv2.imwrite(img_name, image_np)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    time.sleep(2)
                                    vs=vs+1
                                    if(vs==5):
                                       vs=0
                                    
                                    
                                    




                        elif ((obje1==obje2=="b'person'" and obje0=="b'bag'") and (mesafe10 > mesafe20)):
                            cv2.circle(image_np, (axort[0], ayort[0]), 10, (0, 0, 255), 5)
                            sonuc_2[k] = int(mesafe10 - mesafe20)
                            k = k + 1
                            if (k > 3):
                                if (((sonuc_2[k - 1] > 0 and sonuc_2[k - 2] < 0)) or ((sonuc_2[k - 1] < 0 and sonuc_2[k - 2] > 0))):
                                    
                                    start = time.time()
                                    dt_string=str(vs)
                                    img_name = "deneme/face/"+ dt_string + ".jpg"
                                    cv2.imwrite(img_name, image_np)
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    resim=cv2.imread("cap.jpg")
                                    cropped=resim[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    resim=cv2.imread("cap.jpg")
                                    cropped=resim[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    time.sleep(2)
                                    vs=vs+1
                                    if(vs==5):
                                       vs=0

                      



                        elif ((obje1 == obje0 == "b'person'" and obje2 == "b'bag'")and (mesafe12 > mesafe20)):
                            cv2.circle(image_np, (axort[2], ayort[2]), 10, (0, 0, 255), 10)
                            sonuc_3[l] = int(mesafe12 - mesafe20)
                            l = l + 1
                            if (l > 3):
                                if (((sonuc_3[l - 1] > 0 and sonuc_3[l - 2] < 0)) or ((sonuc_3[l - 1] < 0 and sonuc_3[l - 2] > 0))):
                                    
                                    start = time.time()
                                    dt_string=str(vs)
                                    img_name = "deneme/face/"+ dt_string + ".jpg"
                                    cv2.imwrite(img_name, image_np)
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    resim=cv2.imread("cap.jpg")
                                    cropped=resim[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    resim=cv2.imread("cap.jpg")
                                    cropped=resim[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    time.sleep(2)
                                    print("TEHLİKEEEEE")
                                    
                                    vs=vs+1
                                    if(vs==5):
                                       vs=0



                                    

                    """if (len(isim) == 4 and len(isim) != 5):
                        obje0 = str(isim[0])
                        obje1 = str(isim[1])
                        obje2 = str(isim[2])
                        obje3 = str(isim[3])

                        for i in range(4):
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
                        farkx30 = abs(axort[3] - axort[0])
                        farkx31 = abs(axort[3] - axort[1])
                        farkx32 = abs(axort[3] - axort[2])

                        farky10 = abs(ayort[1] - ayort[0])
                        farky12 = abs(ayort[1] - ayort[2])
                        farky20 = abs(ayort[2] - ayort[0])
                        farky30 = abs(ayort[3] - ayort[0])
                        farky31 = abs(ayort[3] - ayort[1])
                        farky32 = abs(ayort[3] - ayort[2])

                        mesafe10 = pow((farkx10 * farkx10 + farky10 * farky10), 0.5)
                        mesafe20 = pow((farkx20 * farkx20 + farky20 * farky20), 0.5)
                        mesafe12 = pow((farkx12 * farkx12 + farky12 * farky12), 0.5)
                        mesafe30 = pow((farkx30 * farkx30 + farky30 * farky30), 0.5)
                        mesafe31 = pow((farkx31 * farkx31 + farky31 * farky31), 0.5)
                        mesafe32 = pow((farkx32 * farkx32 + farky32 * farky32), 0.5)

                        # print(isim[b], axort[b], ayort[b])

                        if ((obje0 == obje2 == obje3== "b'person'" and obje1 == "b'bag'") and (mesafe31>=mesafe10 > mesafe12)):
                            # print("canta name3 e yakin")
                            cv2.circle(image_np, (axort[1], ayort[1]), 10, (0, 0, 255), 5)
                            sonuc_1[j] = int(mesafe10 - mesafe12)
                            j = j + 1
                            if (j > 3):
                                if (((sonuc_1[j - 1] > 0 and sonuc_1[j - 2] < 0)) or ((sonuc_1[j - 1] < 0 and sonuc_1[j - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje0 == obje2 == obje3== "b'person'" and obje1 == "b'bag'") and (mesafe31 >= mesafe12 > mesafe10)):
                            # print("canta name1e yakin")
                            cv2.circle(image_np, (axort[1], ayort[1]), 10, (0, 0, 255), 5)
                            sonuc_1[j] = int(mesafe10 - mesafe12)
                            j = j + 1
                            if (j > 3):
                                if (((sonuc_1[j - 1] > 0 and sonuc_1[j - 2] < 0)) or ((sonuc_1[j - 1] < 0 and sonuc_1[j - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje0 == obje2 == obje3== "b'person'" and obje1 == "b'bag'") and (mesafe12 >= mesafe31 > mesafe10)):
                            # print("canta name1e yakin")
                            cv2.circle(image_np, (axort[1], ayort[1]), 10, (0, 0, 255), 5)
                            sonuc_4[m] = int(mesafe10 - mesafe31)
                            m = m + 1
                            if (m > 3):
                                if (((sonuc_4[m - 1] > 0 and sonuc_4[m - 2] < 0)) or ((sonuc_4[m - 1] < 0 and sonuc_4[m - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[3]:ayort[3], axmin[3]:axmax[3]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje0 == obje2 == obje3== "b'person'" and obje1 == "b'bag'") and (mesafe12 >= mesafe10 > mesafe31)):
                            # print("canta name1e yakin")
                            cv2.circle(image_np, (axort[1], ayort[1]), 10, (0, 0, 255), 5)
                            sonuc_4[m] = int(mesafe10 - mesafe31)
                            m = m + 1
                            if (m > 3):
                                if (((sonuc_4[m - 1] > 0 and sonuc_4[m - 2] < 0)) or ((sonuc_4[m - 1] < 0 and sonuc_4[m - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[3]:ayort[3], axmin[3]:axmax[3]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0




                        elif ((obje1 == obje2 == obje3== "b'person'" and obje0 == "b'bag'") and (mesafe30>=mesafe10 > mesafe20)):
                            cv2.circle(image_np, (axort[0], ayort[0]), 10, (0, 0, 255), 5)
                            sonuc_2[k] = int(mesafe10 - mesafe20)
                            k = k + 1
                            if (k > 3):
                                if (((sonuc_2[k - 1] > 0 and sonuc_2[k - 2] < 0)) or ((sonuc_2[k - 1] < 0 and sonuc_2[k - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje1 == obje2 == obje3=="b'person'" and obje0 == "b'bag'") and (mesafe30 >= mesafe20 > mesafe10)):
                            cv2.circle(image_np, (axort[0], ayort[0]), 10, (0, 0, 255), 5)
                            sonuc_2[k] = int(mesafe10 - mesafe20)
                            k = k + 1
                            if (k > 3):
                                if (((sonuc_2[k - 1] > 0 and sonuc_2[k - 2] < 0)) or ((sonuc_2[k - 1] < 0 and sonuc_2[k - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0
                                    


                        elif ((obje1 == obje2 == obje3== "b'person'" and obje0 == "b'bag'") and (mesafe10>=mesafe30 > mesafe20)):
                            cv2.circle(image_np, (axort[0], ayort[0]), 10, (0, 0, 255), 5)
                            sonuc_5[n] = int(mesafe30 - mesafe20)
                            n = n + 1
                            if (n > 3):
                                if (((sonuc_5[n - 1] > 0 and sonuc_5[n - 2] < 0)) or ((sonuc_5[n - 1] < 0 and sonuc_5[n - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[3]:ayort[3], axmin[3]:axmax[3]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje1 == obje2 == obje3=="b'person'" and obje0 == "b'bag'") and (mesafe10 >= mesafe20 > mesafe30)):
                            cv2.circle(image_np, (axort[0], ayort[0]), 10, (0, 0, 255), 5)
                            sonuc_5[n] = int(mesafe30 - mesafe20)
                            n = n + 1
                            if (n > 3):
                                if (((sonuc_5[n - 1] > 0 and sonuc_5[n - 2] < 0)) or ((sonuc_5[n - 1] < 0 and sonuc_5[n - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[3], axmin[3]:axmax[3]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0



                        elif ((obje1 == obje0 == obje3 == "b'person'" and obje2 == "b'bag'") and (mesafe32 >= mesafe12 > mesafe20)):
                            cv2.circle(image_np, (axort[2], ayort[2]), 10, (0, 0, 255), 10)
                            sonuc_3[l] = int(mesafe12 - mesafe20)
                            l = l + 1
                            if (l > 3):
                                if (((sonuc_3[l - 1] > 0 and sonuc_3[l - 2] < 0)) or ((sonuc_3[l - 1] < 0 and sonuc_3[l - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0



                        elif ((obje1 == obje0 == obje3== "b'person'" and obje2 == "b'bag'") and (mesafe32 >= mesafe20 > mesafe12)):
                            cv2.circle(image_np, (axort[2], ayort[2]), 10, (0, 0, 255), 10)
                            sonuc_3[l] = int(mesafe12 - mesafe20)
                            l = l + 1
                            if (l > 3):
                                if (((sonuc_3[l - 1] > 0 and sonuc_3[l - 2] < 0)) or ((sonuc_3[l - 1] < 0 and sonuc_3[l - 2] > 0)) and (time.time() - start > 2)):
                                   
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje1 == obje0 == obje3 == "b'person'" and obje2 == "b'bag'") and (mesafe12 >= mesafe32 > mesafe20)):
                            cv2.circle(image_np, (axort[2], ayort[2]), 10, (0, 0, 255), 10)
                            sonuc_6[o] = int(mesafe32 - mesafe20)
                            o = o + 1
                            if (o > 3):
                                if (((sonuc_6[o - 1] > 0 and sonuc_6[o - 2] < 0)) or ((sonuc_6[o - 1] < 0 and sonuc_6[o - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[3]:ayort[3], axmin[3]:axmax[3]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0



                        elif ((obje1 == obje0 == obje3== "b'person'" and obje2 == "b'bag'") and (mesafe12 >= mesafe20 > mesafe32)):
                            cv2.circle(image_np, (axort[2], ayort[2]), 10, (0, 0, 255), 10)
                            sonuc_6[o] = int(mesafe32 - mesafe20)
                            o = o + 1
                            if (o > 3):
                                if (((sonuc_6[o - 1] > 0 and sonuc_6[o - 2] < 0)) or ((sonuc_6[o - 1] < 0 and sonuc_6[o - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[3]:ayort[3], axmin[3]:axmax[3]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                   
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje1 == obje0 == obje2== "b'person'" and obje3 == "b'bag'") and (mesafe31 >= mesafe30 > mesafe32)):
                            cv2.circle(image_np, (axort[3], ayort[3]), 10, (0, 0, 255), 10)
                            sonuc_7[p] = int(mesafe30 - mesafe32)
                            p = p + 1
                            if (p > 3):
                                if (((sonuc_7[p - 1] > 0 and sonuc_7[p - 2] < 0)) or ((sonuc_7[p - 1] < 0 and sonuc_7[p - 2] > 0)) and (time.time() - start > 2)):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje1 == obje0 == obje2== "b'person'" and obje3 == "b'bag'") and (mesafe31 >= mesafe32 > mesafe30)):
                            cv2.circle(image_np, (axort[3], ayort[3]), 10, (0, 0, 255), 10)
                            sonuc_7[p] = int(mesafe30 - mesafe32)
                            l = l + 1
                            p = p + 1
                            if (p > 3):
                                if (((sonuc_7[p - 1] > 0 and sonuc_7[p - 2] < 0)) or ((sonuc_7[p - 1] < 0 and sonuc_7[p - 2] > 0))):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje1 == obje0 == obje2 == "b'person'" and obje3 == "b'bag'") and (
                                mesafe30 >= mesafe31 > mesafe32)):
                            cv2.circle(image_np, (axort[3], ayort[3]), 10, (0, 0, 255), 10)
                            sonuc_8[r] = int(mesafe31 - mesafe32)
                            r = r + 1
                            if (r > 3):
                                if (((sonuc_8[r - 1] > 0 and sonuc_8[r - 2] < 0)) or ((sonuc_8[r - 1] < 0 and sonuc_8[r - 2] > 0))):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje1 == obje0 == obje2 == "b'person'" and obje3 == "b'bag'") and (
                                mesafe30 >= mesafe32 > mesafe31)):
                            cv2.circle(image_np, (axort[3], ayort[3]), 10, (0, 0, 255), 10)
                            sonuc_8[r] = int(mesafe31 - mesafe32)
                            r = r + 1
                            if (r > 3):
                                if (((sonuc_8[r - 1] > 0 and sonuc_8[r - 2] < 0)) or ((sonuc_8[r - 1] < 0 and sonuc_8[r - 2] > 0))):
                                    
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[2]:ayort[2], axmin[2]:axmax[2]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje1 == obje0 == obje2 == "b'person'" and obje3 == "b'bag'") and (
                                mesafe32 >= mesafe31 > mesafe30)):
                            cv2.circle(image_np, (axort[3], ayort[3]), 10, (0, 0, 255), 10)
                            sonuc_9[s] = int(mesafe31 - mesafe30)
                            s = s + 1
                            if (s > 3):
                                if (((sonuc_9[s - 1] > 0 and sonuc_9[s - 2] < 0)) or ((sonuc_9[s - 1] < 0 and sonuc_9[s - 2] > 0))):
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0

                        elif ((obje1 == obje0 == obje2 == "b'person'" and obje3 == "b'bag'") and (
                                mesafe32 >= mesafe30 > mesafe31)):
                            cv2.circle(image_np, (axort[3], ayort[3]), 10, (0, 0, 255), 10)
                            sonuc_9[s] = int(mesafe31 - mesafe30)
                            s = s + 1
                            if (s > 3):
                                if (((sonuc_9[s - 1] > 0 and sonuc_9[s - 2] < 0)) or ((sonuc_9[s - 1] < 0 and sonuc_9[s - 2] > 0))):
                                    start = time.time()
                                    dt_string=str(vs);
                                    img_name1 = "deneme/face/"+ dt_string + "1.jpg"
                                    cropped=image_np[aymin[0]:ayort[0], axmin[0]:axmax[0]]
                                    cv2.imwrite(img_name1, cropped)
                                    
                                    img_name2 = "deneme/face/"+ dt_string + "2.jpg"
                                    cropped=image_np[aymin[1]:ayort[1], axmin[1]:axmax[1]]
                                    cv2.imwrite(img_name2, cropped)
                                    bot.send_message(chatid, "Tehlikeli Bir Durum Olabilir!!!")
                                    bot.send_photo(chat_id=chatid, photo=open(img_name, 'rb'), timeout=100)
                                    print("TEHLİKEEEEE")
                                    vs=vs+1
                                    if(vs==7):
                                       vs=0"""


                        
    


            #name = str(isim)
            #print(objects)
            #print(bool(objects))
            #print(name)
            #if bool(isim)==True:
            #print (isim[0])
            resized = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)
            #cv2.imshow('object detection', image_np)
            img_local="imshow/"+"image_np.jpg"
            cv2.imwrite(img_local, image_np)
            dosya2=open('q.txt','r+',encoding="utf-8")
            dizis=dosya2.readlines()
            kapa=dizis[1]
            dosya2.close()
            
            
            #cv2.destroyAllWindows()
        
        # pfps = time.time() - fps
        # print("Fps: "+ str(int(1/pfps)))
    # Clean up
    
    
            if ((cv2.waitKey(25) & 0xFF == ord('q'))or kapa=='q\n'):
                cv2.destroyAllWindows()
                cap.release()
                break
