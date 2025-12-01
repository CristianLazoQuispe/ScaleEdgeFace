import sys
import os

local_path = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(local_path,'../'))

def get_model_face_detection(model_name,**kwargs):
    print("MODEL INPUT : ",model_name)
    if model_name == "mediapipe":
        from models.face_detection.mediapipe import model_mediapipe
        model =  model_mediapipe.MODEL_mediapipe()
    elif model_name == "kornia":
        from models.face_detection.kornia_face import model_kornia_face
        model = model_kornia_face.MODEL_kornia()
    elif model_name == "yolov8face":
        from models.face_detection.yolov8face import model_yolo8face
        model =   model_yolo8face.YOLOv8_face("yolov8n-face.onnx")
    elif model_name == "faceboxes_tensorrt":
        from models.face_detection.faceboxes.faceboxes_trt import FaceBoxes_trt
        model  = FaceBoxes_trt.FaceDetector("FaceBoxesProdTrt.pb",device='gpu',gpu_memory_fraction=0.1)
    elif model_name == "faceboxes_tensorrt_FP16":
        from models.face_detection.faceboxes.faceboxes_trt import FaceBoxes_trt
        model  = FaceBoxes_trt.FaceDetector("trt/trt_faceboxes_precision_FP16_segment_50_work_10.pb",device='gpu',gpu_memory_fraction=0.1)
    elif model_name == "faceboxes_tensorrt_FP32":
        from models.face_detection.faceboxes.faceboxes_trt import FaceBoxes_trt
        model  = FaceBoxes_trt.FaceDetector("trt/trt_faceboxes_precision_FP32_segment_50_work_10.pb",device='gpu',gpu_memory_fraction=0.1)
    elif model_name == "faceboxes_tensorrt_INT8":
        from models.face_detection.faceboxes.faceboxes_trt import FaceBoxes_trt
        model  = FaceBoxes_trt.FaceDetector("trt/trt_faceboxes_precision_INT8_segment_50_work_10.pb",device='gpu',gpu_memory_fraction=0.1)
    elif model_name == "faceboxes_torch":
        from models.face_detection.faceboxes.faceboxes_torch import FaceBoxes_torch
        model = FaceBoxes_torch.FaceBoxes(timer_flag=False)
    elif model_name == "faceboxes_tensorflow_gpu":
        from models.face_detection.faceboxes.faceboxes_tensorflow import FaceBoxes_tensorflow
        model  = FaceBoxes_tensorflow.FaceDetector(device='gpu',gpu_memory_fraction=0.3)
    elif model_name == "faceboxes_tensorflow_cpu":
        from models.face_detection.faceboxes.faceboxes_tensorflow import FaceBoxes_tensorflow
        model  = FaceBoxes_tensorflow.FaceDetector(device='cpu',gpu_memory_fraction=0.3)
    elif model_name == "faceboxes_onnx":
        from models.face_detection.faceboxes.faceboxes_onnx import FaceBoxes_onnx
        model = FaceBoxes_onnx.FaceBoxes_ONNX(timer_flag=False)
    else:
        print("MODEL LOADED : ","mediapipe")
        from models.face_detection.mediapipe import model_mediapipe
        return model_mediapipe.MODEL_mediapipe()
    
    print("MODEL LOADED : ",model_name)
    return model


def get_model_tracking(model_name,**kwargs):
    print("MODEL INPUT : ",model_name)

    if model_name == "Norfair":
        from models.face_tracking import trackerNorfair
        tracker = trackerNorfair.Tracker(distance_threshold=200,hit_counter_max=5,period=2)
    elif model_name == "Sort":
        from models.face_tracking import trackerSort
        tracker = trackerSort.Tracker(max_age=1,min_hits=10)
    else:
        print("MODEL LOADED : ","Norfair")
        from models.face_tracking import trackerNorfair
        return trackerNorfair.Tracker(distance_threshold=200,hit_counter_max=5,period=2)
    
    print("MODEL LOADED : ",model_name)
    return tracker


def get_model_face_recognition(model_name,**kwargs):
    print("MODEL INPUT : ",model_name)

    if model_name == "face_recognition":
        import face_recognition
        model = face_recognition
    else:
        print("MODEL LOADED : ","face_recognition")
        import face_recognition
        return face_recognition
    
    print("MODEL LOADED : ",model_name)
    return model



