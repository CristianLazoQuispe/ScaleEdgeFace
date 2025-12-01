import os
import time
import logging
import numpy as np

class FPS_logging:
    def __init__(self, max_size: int = 15, args=None, prefix="dev", device="computer") -> None:
        self.list_time = []
        self.max_size = max_size
        self.device = device

        if args:
            video_name = os.path.splitext(os.path.basename(args.camera_src))[0]
            facedetection_model = args.facedetection_model if args.facedetection_use else ""
            facerecognition_model = args.facerecognition_model if args.facerecognition_use else ""
            tracking_model = args.facetracking_model if args.facetracking_use else ""

            name_file = prefix + "_" + str(video_name) + '_' + str(args.camera_width) + '_' + str(
                args.camera_height)
            name_file = name_file + '_' + facedetection_model + '_' + tracking_model + '_' + facerecognition_model

            self.name = os.path.join('results/output_time', self.device, name_file + '.log')

            self.create_folder(os.path.join('results/output_time', self.device))

            logging.basicConfig(filename=self.name, level=logging.INFO)
            logging.info(str(args))

    def create_folder(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def start(self):
        self.time_start = time.time()

    def close(self):
        logging.shutdown()

    def update(self):
        self.time_end = time.time()
        total_time = self.time_end - self.time_start

        if len(self.list_time) >= self.max_size:
            _ = self.list_time.pop(0)

        self.list_time.append(total_time)

        logging.info(str(total_time))

        return total_time

    def fps(self):
        if not self.list_time:
            return 0

        # self.fps_value = len(self.list_time)/np.sum(self.list_time)
        self.fps_value = 1 / np.percentile(self.list_time, 50)
        return self.fps_value

    def elapsed(self):
        if not self.list_time:
            return 0
        self.elapsed_value = np.sum(self.list_time) / len(self.list_time)
        return self.elapsed_value
    
class FPS:
    def __init__(self,max_size :int=15,args=None,prefix="dev",device="computer") -> None:
        self.list_time = []
        self.max_size = max_size
        self.file = None
        self.device = device
        if args:
            
            if args.camera_src.split(".")[-1] in ['avi','mp4']:
                self.video_name = args.camera_src.split("/")[-1].split(".")[0]
            else:
                self.video_name = "webcam_"+args.camera_src
            
            facedetection_model   = args.facedetection_model if args.facedetection_use else ""
            facerecognition_model = args.facerecognition_model if args.facerecognition_use else ""
            tracking_model        = args.facetracking_model      if args.facetracking_use else ""
            
            self.exp_name = args.exp_name
            name_file = prefix+"_"+str(self.video_name) + '_'+str(args.camera_width)+'_'+str(args.camera_height)
            name_file = name_file + '_'+facedetection_model+'_'+tracking_model+  '_'+facerecognition_model
            #+'_'+datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')

            self.filename  = os.path.join('results/output_time',self.device,self.exp_name,self.video_name,name_file+'.txt') 

            self.create_folder( os.path.join('results/output_time',self.device))
            self.create_folder( os.path.join('results/output_time',self.device,self.exp_name))
            self.create_folder( os.path.join('results/output_time',self.device,self.exp_name,self.video_name))
                
            if args.fps_save:
                self.file = open(self.filename, 'w')
                self.file.write(str(args))
                self.file.write('\n')
        
    def create_folder(self,directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    def start(self):
        self.time_start = time.time()
        
    def close(self):
        if self.file !=None:
            self.file.close()

    def update(self):
        self.time_end = time.time()
        total_time = self.time_end-self.time_start
        
        if len(self.list_time)>=self.max_size:
            _ = self.list_time.pop(0)
        
        self.list_time.append(total_time)
        
        if self.file !=None:
            self.file.write(str(total_time))
            self.file.write('\n')

        return total_time
    
    def fps(self):
        if len(self.list_time)==0:
            return 0
        #self.fps_value = len(self.list_time)/np.sum(self.list_time)
        self.fps_value = 1/np.percentile(self.list_time, 50)
        return self.fps_value
    
    def elapsed(self):
        if len(self.list_time)==0:
            return 0
        self.elapsed_value = np.sum(self.list_time)/len(self.list_time)
        return self.elapsed_value