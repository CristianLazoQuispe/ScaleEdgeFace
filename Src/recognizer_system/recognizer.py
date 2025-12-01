import cv2
import numpy as np
import sys
import os

from time import gmtime, strftime
import threading
import matplotlib.pyplot as plt
import time
import random
import json
import collections
local_path = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(local_path,'../'))

from utils import my_logs 
from utils import manager

class ThreadRecognizer(threading.Thread):
    logger = my_logs.Logger("Recognizer")
    def __init__(self,variables,model_recognition):
        threading.Thread.__init__(self)
        ThreadRecognizer.logger.info("Loading Recognizer")
        self.event = threading.Event()
        self.working = False
        self.working_lock = threading.Lock()
        self.update_lock = threading.Lock()
        self.variables = variables
        self.model_recognition = model_recognition
        self.image = None
        self.bboxes_faces = None
        
        ThreadRecognizer.logger.info("Loaded Recognizer")
        
    def stop(self):
        self.event.set()
    def can_work(self):
        can = False
        self.update_lock.acquire()
        can = self.bboxes_faces is not None
        self.update_lock.release()
        return can
    def is_working(self):
        self.working_lock.acquire()
        working = self.working
        self.working_lock.release()
        return working

    def set_working(self, is_working):
        self.working_lock.acquire()
        self.working = is_working
        self.working_lock.release()

    def enqueue(self, image,bboxes_faces):
        if not self.is_working():
            self.update_lock.acquire()
            self.bboxes_faces = bboxes_faces			
            self.image = image
            self.update_lock.release()
    def run(self):
        ThreadRecognizer.logger.info("Starting runing recognize algorithm")
        while not self.event.is_set():
            if self.can_work():
                self.recognize()
            self.event.wait(0.1)
        ThreadRecognizer.logger.info("Finished runing recognize algorithm ")
        
    def recognize(self):
        self.set_working(True)
        self.update_lock.acquire()
        bboxes_faces = self.bboxes_faces.copy()
        image        = self.image.copy()
        self.bboxes_faces = None        
        self.image        = None
        self.update_lock.release()
        ##############################################		        
        #print("HILO :D ",self.variables.id_queue_names)
        #print("bboxes_faces :" ,bboxes_faces," image",image.shape)
        boxes_to_face_recognition  = []
        for [x1, y1, x2, y2,id] in bboxes_faces:
            boxes_to_face_recognition.append([int(y1), int(x2), int(y2), int(x1)])

        
        face_encodings = self.model_recognition.face_encodings(image, boxes_to_face_recognition, num_jitters=-1)#[0]

        for i in range(len(bboxes_faces)):
            id_face = bboxes_faces[i][4]
            face_encoding = face_encodings[i]
            face_distances = self.model_recognition.face_distance(self.variables.list_embeddings, face_encoding)
            index_min      = np.argpartition(face_distances,2)
            min_distance       = float(face_distances[index_min[0]])
            if min_distance<0.5:
                name = self.variables.list_names[index_min[0]]
                
                if id_face in self.variables.id_queue_names.keys():
                    self.variables.id_queue_names[id_face].append(name)
                else:
                    self.variables.id_queue_names[id_face]=[name]
                
                counter = collections.Counter(self.variables.id_queue_names[id_face])
                most_commons = counter.most_common(1)
                first_name,first_count = most_commons[0]
                
                if first_count == self.variables.limit:
                    self.variables.id_to_names[id_face] = first_name
                    self.variables.id_to_color[id_face] = (255,0,0)
                
        self.set_working(False)


class variables_Recognizer(object):
    def __init__(self,path_database="results/dict_faces.json",limit=2) -> None:     
        self.path_database = path_database
        self.limit = limit
        #self.ids_recognized_cnt = {}
        self.id_to_names = {}
        self.id_queue_names = {}
        self.id_to_color = {}
        self.list_embeddings = None
        self.list_names = None
        self.read_database(self.path_database)
        
    def clean(self):
        #self.ids_recognized_cnt = {}
        self.id_to_names = {}
        self.id_to_color = {}
        self.id_queue_names = {}

    def read_database(self,path_database):
        with open(path_database, "r") as outfile:
            # Reading from json file
            json_object = json.load(outfile)
    
        list_names = []
        list_embeddings = []

        for name in json_object.keys():
            embeddings = json_object[name]
            for embedding in embeddings:
                list_names.append(name)
                list_embeddings.append(embedding)

        self.list_names      = np.array(list_names)
        self.list_embeddings = np.array(list_embeddings)

class Recognizer(object):
    
    def __init__(self,model_recognition=None,limit = 2,path_database="results/dict_faces.json") -> None:
        
        if model_recognition is None:
            model_recognition = manager.get_model_face_recognition("face_recognition")
        self.limit = limit
        self.variables         =  variables_Recognizer(limit=limit)
        self.thread_recognizer =  ThreadRecognizer(self.variables,model_recognition)

    def start(self):
        self.thread_recognizer.start()
    
    def stop(self):
        self.thread_recognizer.event.set()
    
    def who_is_unrecognized(self,boxes_tracker):
        boxes_unrecognized = []
        for box in boxes_tracker:
            id_local = box[4]
            #if id_local in self.variables.ids_recognized_cnt.keys():
            #    if self.variables.ids_recognized_cnt[id_local]<self.limit:
            #        boxes_unrecognized.append(box)
            #else:
            #    boxes_unrecognized.append(box)       
            if not id_local in self.variables.id_to_names.keys():
                boxes_unrecognized.append(box)       
        return boxes_unrecognized
    
    def run_thread(self,image,boxes_tracker):
        #print("boxes_tracker : ",boxes_tracker)
        #print("self.variables.id_to_names : ",self.variables.id_to_names)
        boxes_unrecognized = self.who_is_unrecognized(boxes_tracker)
        #print("boxes_unrecognized : ",boxes_unrecognized)        
        if not self.thread_recognizer.is_working() and len(boxes_unrecognized)>0:
            #print("lanzar hilo")
            self.thread_recognizer.enqueue(image,boxes_unrecognized)
    
    def get_names_color(self,boxes_tracker):
        names = []
        colors = []
        for box in boxes_tracker:
            id_local = box[4]
            if id_local in self.variables.id_to_names.keys():
                names.append(self.variables.id_to_names[id_local])
                colors.append(self.variables.id_to_color[id_local])
            else:
                names.append("DESCONOCIDO")
                colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
        return names,colors
    
                    
    def predict(self,image,boxes_tracker):
        if len(boxes_tracker)==0:
            self.variables.clean()
            return [],[]
        
        self.run_thread(image,boxes_tracker)
        
        return self.get_names_color(boxes_tracker)
