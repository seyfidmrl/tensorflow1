import tensorflow as tf
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=500)
parser.add_argument('--cam_height', type=int, default=500)
parser.add_argument('--scale_factor', type=float, default=1) #0.7125
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()



with tf.Session() as sess:
    model_cfg, model_outputs = posenet.load_model(args.model, sess)
    output_stride = model_cfg['output_stride']
    

    if args.file is not None:
        cap = cv2.VideoCapture(args.file)
    else:
        cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width) #640-480
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0
    
    
    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)   
    
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
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
        
       
        b= []
        
        
        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image, b = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords, b,
            min_pose_score=0.15, min_part_score=0.1)
        
        
        nose = [None]*10
        bilek_sal = [None]*10
        bilek_sol = [None]*10 
        #print(b)
        
        
        for i in range(len(b)):
            if b[i] == []:
                continue
            for m in range(len(b[i])):
                
                if b[i][m][0] == 9:
                    bilek_sol[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                    print(i)
                    print("bilek_sol")
                    print( bilek_sol[i])
                    
                if b[i][m][0] == 10:
                    bilek_sal[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                    print(i)
                    print("bilek_sal")
                    print( bilek_sal[i])
                
                if b[i][m][0] == 1:
                    nose[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                   
                
        
        # for i in range(len(nose)):   
            # if nose[i] != None and left_eye[i] !=  None  and right_eye[i] !=  None:
                # fark = abs(left_eye[i][0]-right_eye[i][0])
                # start_point = (int(left_eye[i][0] - fark), nose[i][1] + int(fark*1.5))
                # end_point= (int(right_eye[i][0] + fark), nose[i][1] - int(fark*2.5))                
                # overlay_image = cv2.rectangle(overlay_image, start_point, end_point, color=(0,250,0), thickness=5) 
                
        
        cv2.imshow('posenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print('Average FPS: ', frame_count / (time.time() - start))
    cap.release()
    cv2.destroyAllWindows()
