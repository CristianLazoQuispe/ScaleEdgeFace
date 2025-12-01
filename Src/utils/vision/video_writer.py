import os
import cv2
import time
from queue import Queue
from threading import Thread


class ParallelVideoWriter:
    def __init__(self, width, height, fps=30,args=None,prefix="dev",device="computer",name_file=None):
        #self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.out = None
        self.thread = None
        self.device = device
        self.queue = []
        self.stop_thread = False

        if args:
            
            if args.camera_src.split(".")[-1] in ['avi','mp4']:
                self.video_name = args.camera_src.split("/")[-1].split(".")[0]
            else:
                self.video_name = "webcam_"+args.camera_src
            
            facedetection_model   = args.facedetection_model if args.facedetection_use else ""
            facerecognition_model = args.facerecognition_model if args.facerecognition_use else ""
            tracking_model        = args.facetracking_model      if args.facetracking_use else ""
            
            self.exp_name = args.exp_name
            if name_file is None:                
                name_file = prefix+"_"+str(self.video_name) + '_'+str(args.camera_width)+'_'+str(args.camera_height)
                name_file = name_file + '_'+facedetection_model+'_'+tracking_model+  '_'+facerecognition_model+".avi"
            
            #+'_'+datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')
            self.filename  = os.path.join('results/output_time',self.device,self.exp_name,self.video_name,name_file) 

            self.create_folder( os.path.join('results/output_time',self.device))
            self.create_folder( os.path.join('results/output_time',self.device,self.exp_name))
            self.create_folder( os.path.join('results/output_time',self.device,self.exp_name,self.video_name))

        self.last_frame_time = time.time()
        self.frame_queue = Queue()
                
    def create_folder(self,directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    def start(self):
        self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.width, self.height))
        print("MJPG")
        #self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'FFV1'), self.fps, (self.width, self.height))
        self.thread = Thread(target=self._write_frames)
        self.thread.start()
        
    def update(self, frame):
        if self.out is not None:
            current_time = time.time()
            time_elapsed = current_time - self.last_frame_time
            time_between_frames = 1 / self.fps

            if time_elapsed >= time_between_frames:
                self.frame_queue.put(frame)
                self.last_frame_time = current_time

    def close(self):
        self.stop_thread = True  # Indica al hilo que debe finalizar
        if self.thread is not None:
            self.thread.join()
        time.sleep(1)
        if self.out is not None:
            self.out.release()

    def _write_frames(self):
        while not self.stop_thread:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.out.write(frame)
            else:
                time.sleep(0.001)  # Evitar el polling constante sin trabajo
