import sys
import os
import gc
gc.collect()
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}

import sys
import cv2
import numpy as np
import traceback

sys.path.insert(1, '../../Src/')

from utils import vision
from utils import camera

COLOR_BOUNDINGBOX = (255, 0, 0)
THICKNESS_BOUNDINGBOX = 2

from models.kornia_face import model_kornia_face
model = model_kornia_face.MODEL_kornia()



cap = camera.ParallelCamera(0).start()
my_fps = vision.FPS(100)
my_fps_model = vision.FPS(100)

try:

  while True:
    my_fps.start()

    # Read image
    success, image = cap.read()  
    if not success:
      print("Ignoring empty camera frame.")
      break

    image = cv2.flip(image.copy(), 1)
    image = cv2.resize(image,(300,300))

    my_fps_model.start()  
    # Model Inference
    image.flags.writeable = False
    boxes, scores= model.predict(image)
    image.flags.writeable = True  
    
    # Draw bounding boxes
    image = vision.draw_boundingboxes(image,boxes,COLOR_BOUNDINGBOX,THICKNESS_BOUNDINGBOX,if_wh=False)
    
    # Calculate FPS
    my_fps_model.update()
    my_fps.update()
    # Show image
    image = vision.put_text(image,"FPS      : "+str(np.round(my_fps.fps(),3)),pos =(20,20))
    image = vision.put_text(image,"FPS model: "+str(np.round(my_fps_model.fps(),3)),pos =(20,50))
    cv2.imshow('Face Detection Project',image)
  
    # Close window with ESC
    if cv2.waitKey(5) & 0xFF == 27:
      break

except Exception as e:
    cap.release()
    cv2.destroyAllWindows()
    import sys
    sys.exc_info()
    print(traceback.format_exc())  # or: traceback.print_exc()
    
    
# DestroyWindows
cap.release()
cv2.destroyAllWindows()
