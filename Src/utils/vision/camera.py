import cv2
from threading import Thread, Lock
import numpy as np
import time

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports
  



 
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



class ParallelCamera(object):
    def __init__(self, src=0,is_picam = False,height=480,width=640,fps=60):
        self.is_video = False
        self.time_fps = 1.0/fps
        self.height = height
        self.width = width

        if is_picam:
            print("start reading picam ...")
            self.stream = cv2.VideoCapture(gstreamer_pipeline(sensor_id=src,flip_method=4,display_height=height,display_width=width),cv2.CAP_GSTREAMER)
            print("start reading picam readed!")
        else:
            print("type id",type(src))
            print("start webcam id: "+str(src)+"...")
            if src in [str(i) for i in range(-2,5)]:
                src = int(src)
            self.stream = cv2.VideoCapture(src)   
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)#OpenCVManager.OUTPUT_SIZE_WIDTH)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,height)# OpenCVManager.OUTPUT_SIZE_HEIGHT)
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            print("start webcam readed!")
            if type(src) == str:
                if src.split(".")[-1] in ['avi','mp4']:
                    self.is_video = True
                
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def first_frame(self):
        (grabbed1, frame1) = self.stream.read()
        frame1 = cv2.resize(frame1, (self.width, self.height))
        return grabbed1,frame1

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        #while not self.thread.stopped():
        while self.started:
            (grabbed, frame) = self.stream.read()
            frame = cv2.resize(frame, (self.width, self.height))
            if self.is_video:
                time.sleep(self.time_fps)
            self.read_lock.acquire()
            self.grabbed = bool(grabbed)
            if grabbed:
                self.frame = frame.copy()
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()        
        grabbed = bool(self.grabbed)
        frame =None
        if grabbed:
            frame = self.frame.copy()
        self.read_lock.release()
        return grabbed,frame

    def release(self):
        self.started = False
        self.thread.join()
        time.sleep(1)
        self.stream.release()

    def __exit__(self, exc_type, exc_value, traceback):
        self.started = False
        self.thread.join()
        time.sleep(1)
        self.stream.release()
