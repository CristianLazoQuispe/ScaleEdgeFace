import mediapipe as mp
import cv2
import gc

class MODEL_mediapipe:
  
  def __init__(self,model_selection=0,min_detection_confidence=0.3) -> None:
    self.model = mp.solutions.face_detection.FaceDetection( min_detection_confidence=min_detection_confidence)

  def predict(self,image_bgr):
    image_bgr.flags.writeable = False
    #image.flags.writeable = False
    #image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = self.model.process(image_bgr)
    #image.flags.writeable = True
    predictions = []
    scores      = []
    if results.detections:
      for detection in results.detections:
        relative_bounding_box = detection.location_data.relative_bounding_box
        score = detection.score
        xmin   = int(relative_bounding_box.xmin*image_bgr.shape[1])
        ymin   = int(relative_bounding_box.ymin*image_bgr.shape[0])
        width  = int(relative_bounding_box.width*image_bgr.shape[1])
        height = int(relative_bounding_box.height*image_bgr.shape[0])
        xmax = xmin+width
        ymax = ymin+height
        predictions.append([xmin,ymin,xmax,ymax])
        scores.append(score)
        
    image_bgr.flags.writeable = True  
    return predictions,scores
  
  
  def close(self):
    del self.model
    gc.collect()
    