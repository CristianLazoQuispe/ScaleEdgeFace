# Python program to illustrate 
# saving an operated video
  
# organize imports
import numpy as np
import cv2
  
# This will return video from the first webcam on your computer.
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)#OpenCVManager.OUTPUT_SIZE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)# OpenCVManager.OUTPUT_SIZE_HEIGHT)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('results/videos/jetson_a02_video_01.avi', fourcc, 60.0, (1280, 720))
  
# loop runs if capturing has been initialized. 
while(True):
    # reads frames from a camera 
    # ret checks return at each frame
    ret, frame = cap.read() 
    # output the frame
    out.write(frame) 
      
    # The original input frame is shown in the window 
    cv2.imshow('Original', frame)
      
    # Wait for 'a' key to stop the program 
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
  
# Close the window / Release webcam
cap.release()
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()