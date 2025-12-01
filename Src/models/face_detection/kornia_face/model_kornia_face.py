import cv2
import gc
import numpy as np
import torch

import kornia as K
from kornia.contrib import FaceDetector, FaceDetectorResult, FaceKeypoint

#dtype = torch.float32

class MODEL_kornia:
  
  def __init__(self,vis_threshold=0.8) -> None:
    # select the device
    self.device = torch.device('cpu')
    if torch.cuda.is_available():
        print("using CUDA")
        self.device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    # create the detector object
    self.model = FaceDetector().to(self.device)
    self.vis_threshold = vis_threshold

  def draw_keypoint(self,img: np.ndarray, det: FaceDetectorResult, kpt_type: FaceKeypoint) -> np.ndarray:
    kpt = det.get_keypoint(kpt_type).int().tolist()
    return cv2.circle(img, kpt, 2, (255, 0, 0), 2)


  def scale_image(self,img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 1.0 * size / w
    return cv2.resize(img, (int(w * scale), int(h * scale)))


  def predict(self,image_bgr):
    image_bgr.flags.writeable = False
    
    #results = self.model.process(image_bgr)
    #image_bgr = self.scale_image(image_bgr,320)
    img = K.image_to_tensor(image_bgr, keepdim=False).to(self.device)
    img = K.color.bgr_to_rgb(img.float())
    # detect !
    with torch.no_grad():
        dets = self.model(img)
        

    dets = [FaceDetectorResult(o) for o in dets[0]]

    #image.flags.writeable = True
    predictions = []
    scores      = []

    for b in dets:
      if b.score < self.vis_threshold:
          continue
      x1, y1 = b.top_left.int().tolist()
      x2, y2 = b.bottom_right.int().tolist()
      score = b.score
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
    