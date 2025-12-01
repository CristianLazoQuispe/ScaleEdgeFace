import cv2
import gc
import numpy as np
import torch

from deepface import DeepFace


class MODEL_deepface:
  
  def __init__(self,model_name,vis_threshold=0.8) -> None:
    """
      model_name : {
        'opencv', 
        'ssd', 
        'dlib', 
        'mtcnn', 
        'retinaface', 
        'mediapipe',
        'yolov8',
        'yunet'
        }
    """
    self.model_name = model_name

    self.vis_threshold = vis_threshold

  def predict(self,image_bgr):
    image_bgr.flags.writeable = False
    
    face_objs = DeepFace.extract_faces(img_path = image_bgr, 
            target_size = (100, 100), 
            enforce_detection=False,
            detector_backend = self.model_name,
            align = False,
    )
    
    #image.flags.writeable = True
    predictions = []
    scores      = []

    for b in face_objs:
      score = b['confidence']
      facial_area = b['facial_area']
      x1 = facial_area['x']
      y1 = facial_area['y']
      x2 = x1+facial_area['w']
      y2 = y1+facial_area['h']

      if score < self.vis_threshold:
          continue
      x1   = int(x1)
      y1   = int(y1)
      x2  = int(x2)
      y2 = int(y2)
      predictions.append([x1,y1,x2,y2])
      scores.append(score)
        
    image_bgr.flags.writeable = True  
    return predictions,scores
  
  
  def close(self):
    del self.model
    gc.collect()
    