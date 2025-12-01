import sys
import os
import gc
import cv2
import traceback
import numpy as np

gc.collect()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'

sys.path.insert(1, '../../Src/')
from utils import vision
from utils import camera
from utils import recognizer
from utils import trackerNorfair,trackerSort
from models.mediapipe import model_mediapipe


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--facedetection_model', help='model facedetection name', type=str, default="mediapipe", nargs='?')
parser.add_argument('--facedetection_use',   help='flag facedetection', type=int, default=1, nargs='?')

parser.add_argument('--tracking_model', help='model tracking name', type=str, default="norfair", nargs='?')
parser.add_argument('--tracking_use',   help='flag tracking', type=int, default=1, nargs='?')

parser.add_argument('--facerecognition_model', help='model facerecognition name', type=str, default="face_recognition", nargs='?')
parser.add_argument('--facerecognition_use',   help='flag facerecognition', type=int, default=1, nargs='?')
parser.add_argument('--facerecognition_limit', help='limit facerecognition', type=int, default=2, nargs='?')
parser.add_argument('--facerecognition_database', help='database name', type=str, default="results/dict_faces.json", nargs='?')

parser.add_argument('--boundingbox_color', help='boundingbox_color ', type=tuple, default=(255, 0, 0), nargs='?')
parser.add_argument('--boundingbox_thickness', help='boundingbox_thickness ', type=int, default=2, nargs='?')

parser.add_argument('--fps_counter', help='fps_counter', type=int, default=100, nargs='?')
parser.add_argument('--fps_precision', help='fps_precision', type=int, default=3, nargs='?')

parser.add_argument('--n_camera', help='n_camera', type=int, default=1, nargs='?')

args = parser.parse_args()

print(args)



recognizer = recognizer.Recognizer(limit = args.facerecognition_limit,path_database=args.facerecognition_database)
model = model_mediapipe.MODEL_mediapipe()
#tracker = trackerSort.Tracker()
tracker = trackerNorfair.Tracker()

   

recognizer.start()
cap = camera.ParallelCamera(args.n_camera).start()

my_fps_total       = vision.FPS(args.fps_counter)
my_fps_tracking    = vision.FPS(args.fps_counter)
my_fps_recognition = vision.FPS(args.fps_counter)
my_fps_detection   = vision.FPS(args.fps_counter)


try:
    while True:
        my_fps_total.start()
        # Read image
        success, image = cap.read()  
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Preprocessing
        #image = cv2.resize(image,(500,500))
        #image = cv2.flip(image.copy(), 1)
        #image = np.array(np.flip(image,1),dtype=np.uint8)


        # Model Detection
        my_fps_detection.start()  
        boxes, scores= model.predict(image)
        my_fps_detection.update()

        # Model Tracking
        my_fps_tracking.start()
        boxes_tracker = tracker.predict(boxes)
        my_fps_tracking.update()

        # Model Recognizer
        my_fps_recognition.start()
        boxes_names,boxes_colors = recognizer.predict(image,boxes_tracker)
        my_fps_recognition.update()
            
        # Drawing bbox    
        #image = vision.draw_boundingboxes(image,boxes_tracker,args.boundingbox_color,args.boundingbox_thickness,if_wh=False,is_track=True)
        image = vision.draw_boxes_names(image,boxes_tracker,boxes_names,boxes_colors,args.boundingbox_thickness,font_scale=1)

        # Calculate FPS total
        my_fps_total.update()

        # Show image
        image = vision.put_text(image,"FPS total : "+str(np.round(my_fps_total.fps(),args.fps_precision)),pos =(10,10))
        image = vision.put_text(image,"FPS model: "+str(np.round(my_fps_detection.fps(),args.fps_precision)),pos =(10,25))
        cv2.imshow('Face Detection Project',image)
        gc.collect()

        # Close window with ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

except Exception as e:
    cap.release()
    recognizer.stop()
    cv2.destroyAllWindows()
    import sys
    sys.exc_info()
    print(traceback.format_exc())  # or: traceback.print_exc()
    
    
# DestroyWindows and clean memory
cap.release()
cv2.destroyAllWindows()
model.close()
recognizer.stop()
