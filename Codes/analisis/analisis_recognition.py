import sys
import os
import gc
import cv2
import time
import traceback
import numpy as np

gc.collect()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'

sys.path.insert(1, '../../Src/')
from utils import vision,camera,manager,clean_memory
from recognizer_system import recognizer

import argparse,configparser

PROJECT_WANDB = "tesis_uni"
ENTITY = "ml_projects"

from dotenv import load_dotenv
import os
import wandb
load_dotenv()

os.environ["WANDB_API_KEY"] =  os.getenv("WANDB_API_KEY")

clean_memory.clean_memory()

parser = argparse.ArgumentParser()
parser.add_argument('--facedetection_model', help='model facedetection name', type=str, default="mediapipe", nargs='?')
parser.add_argument('--facedetection_use',   help='flag facedetection', type=int, default=0, nargs='?')

parser.add_argument('--facetracking_model', help='model tracking name', type=str, default="norfair", nargs='?')
parser.add_argument('--facetracking_use',   help='flag tracking', type=int, default=0, nargs='?')

parser.add_argument('--facerecognition_model', help='model facerecognition name', type=str, default="face_recognition", nargs='?')
parser.add_argument('--facerecognition_use',   help='flag facerecognition', type=int, default=0, nargs='?')
parser.add_argument('--facerecognition_limit', help='limit facerecognition', type=int, default=2, nargs='?')
parser.add_argument('--facerecognition_database', help='database name', type=str, default="results/dict_faces.json", nargs='?')

parser.add_argument('--boundingbox_color_b', help='boundingbox_color_b ', type=int, default=255, nargs='?')
parser.add_argument('--boundingbox_color_g', help='boundingbox_color_g ', type=int, default=0, nargs='?')
parser.add_argument('--boundingbox_color_r', help='boundingbox_color_r ', type=int, default=0, nargs='?')
parser.add_argument('--boundingbox_thickness', help='boundingbox_thickness ', type=int, default=2, nargs='?')

parser.add_argument('--fps_counter', help='fps_counter', type=int, default=100, nargs='?')
parser.add_argument('--fps_precision', help='fps_precision', type=int, default=3, nargs='?')
parser.add_argument('--fps_save', help='fps_save', type=int, default=0, nargs='?')

parser.add_argument('--camera_src', help='camera_src', type=str, default='0', nargs='?')
parser.add_argument('--camera_is_picam', help='camera_is_picam', type=int, default=0, nargs='?')
parser.add_argument('--camera_width', help='camera_width', type=int, default=640, nargs='?')
parser.add_argument('--camera_height', help='camera_height', type=int, default=480, nargs='?')
parser.add_argument('--camera_fps', help='camera_fps', type=int, default=20, nargs='?')
parser.add_argument('--device', help='device of running', type=str, default="computer", nargs='?')

parser.add_argument('--exp_name', help='name of experiment', type=str, default="total_analisis", nargs='?')

parser.add_argument('--video_original_save', help='video_original_save', type=int, default=0, nargs='?')
parser.add_argument('--video_total_save', help='video_total_save', type=int, default=0, nargs='?')
parser.add_argument('--use_wandb', help='use wandb', type=int, default=0, nargs='?')

parser.add_argument('--sweep', help='use sweep', type=int, default=0, nargs='?')

parser.add_argument("-c", "--config_file", type=str, default=None, help='Config file')
#'configs/recognition.conf'
args = parser.parse_args()

if args.config_file:
    print("Loading from .conf file ...")
    config = configparser.ConfigParser()
    config.read(args.config_file)
    defaults = {}
    defaults.update(dict(config.items("Defaults")))
    parser.set_defaults(**defaults)
    args = parser.parse_args() # Overwrite arguments

print(args)

if args.use_wandb:
    run = wandb.init(project=PROJECT_WANDB, 
                        entity=ENTITY,
                        config=args, 
                        name=args.exp_name, 
                        tags=[args.device])


    # Log the parameters to wandb
    #wandb.config.update({
    #    "parameters": total_params,
    #})
    wandb.watch_called = False

boundingbox_color = (args.boundingbox_color_b,args.boundingbox_color_g,args.boundingbox_color_r)


if args.facedetection_use:
    model = manager.get_model_face_detection(args.facedetection_model)
    my_fps_detection   = vision.FPS(args.fps_counter,args,prefix="detection",device=args.device)


if args.facetracking_use:
    tracker = manager.get_model_tracking(args.facetracking_model)
    my_fps_tracking    = vision.FPS(args.fps_counter,args,prefix="tracking",device=args.device)

if args.facerecognition_use:
    model_recognition = manager.get_model_face_recognition(args.facerecognition_model)
    recognizer = recognizer.Recognizer(model_recognition = model_recognition,
                                       limit = args.facerecognition_limit,path_database=args.facerecognition_database)
    recognizer.start()
    my_fps_recognition = vision.FPS(args.fps_counter,args,prefix="recognition",device=args.device)




my_fps_total       = vision.FPS(args.fps_counter,args,prefix="total",device=args.device)

if args.video_total_save:
    video_writer_total = vision.ParallelVideoWriter(width=args.camera_width, height=args.camera_height, fps=30,
                                            args=args,prefix="total",device=args.device,name_file="total.avi")

if args.video_original_save:
    video_writer_original = vision.ParallelVideoWriter(width=args.camera_width, height=args.camera_height, fps=30,
                                            args=args,prefix="total",device=args.device,name_file="original.avi")

fps_detection = None
fps_tracking  = None
fps_recognition = None
fps_total = 30
n_faces = None

try:
    
    cap = camera.ParallelCamera(src=args.camera_src,is_picam=args.camera_is_picam,
                            width=args.camera_width, height=args.camera_height,fps=args.camera_fps)

    print("Reading 2 first frames")
    for i in range(2):
        success, image = cap.first_frame()
        if success:
            if args.facedetection_use:
                boxes, scores= model.predict(image)    
    time.sleep(2)
    
    if args.video_total_save:
        video_writer_total.start()
    if args.video_original_save:
        video_writer_original.start()
    cap.start()
    while True:
        n_faces = 0
        my_fps_total.start()
        # Read image
        success, image = cap.read()  
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Preprocessing
        if args.video_original_save:
            video_writer_original.update(image.copy())

        if args.facedetection_use:
            # Model Detection
            my_fps_detection.start()  
            boxes, scores= model.predict(image)
            fps_detection = my_fps_detection.update()
            n_faces = len(boxes)
            if args.facetracking_use:
                # Model Tracking
                my_fps_tracking.start()
                boxes_tracker = tracker.predict(boxes)
                fps_tracking = my_fps_tracking.update()

                if args.facerecognition_use:
                    # Model Recognizer
                    my_fps_recognition.start()
                    boxes_names,boxes_colors = recognizer.predict(image,boxes_tracker)
                    fps_recognition = my_fps_recognition.update()                        
                    # Drawing bbox    
                    image = vision.draw_boxes_names(image,boxes_tracker,boxes_names,boxes_colors,args.boundingbox_thickness,font_scale=1)
                else:
                    image = vision.draw_boundingboxes(image,boxes_tracker,boundingbox_color,args.boundingbox_thickness,if_wh=False,is_track=True)
            else:
                image = vision.draw_boundingboxes(image,boxes,boundingbox_color,args.boundingbox_thickness,if_wh=False,is_track=False)
                
        # Calculate FPS total
        fps_total = my_fps_total.update()

        # Show image
        total_fps_show = np.round(my_fps_total.fps(),args.fps_precision)
        image_show = vision.put_text(image,"FPS total : "+str(total_fps_show),pos =(10,10))
        cv2.imshow('TESIS CRISTIAN LAZO QUISPE',image_show)
        if args.video_total_save:
            video_writer_total.update(image_show)
        gc.collect()

        if args.use_wandb:
                wandb.log({
                    "time_detection" : fps_detection,
                    "time_tracking"  : fps_tracking,
                    "time_recognition" : fps_recognition,
                    "time_total" : fps_total,
                    "total_fps" : total_fps_show,
                    "n_faces":n_faces
                })
        # Close window with ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

except Exception as e:
    cap.release()
    my_fps_total.close()

    if args.video_total_save:
        video_writer_total.close()
    if args.video_original_save:
        video_writer_original.close()    

    if args.facedetection_use:
        model.close()
        my_fps_detection.close()
        
    if args.facerecognition_use:
        recognizer.stop()
        my_fps_recognition.close()
        
    if args.facetracking_use:
        my_fps_tracking.close()
        
    cv2.destroyAllWindows()
    if args.use_wandb:
        wandb.finish()
    import sys
    sys.exc_info()
    print(traceback.format_exc())  # or: traceback.print_exc()
    
    
my_fps_total.close()

# DestroyWindows and clean memory
cap.release()
cv2.destroyAllWindows()

print("closing models")
if args.facedetection_use:
    model.close()
    my_fps_detection.close()
if args.facerecognition_use:
    recognizer.stop()
    my_fps_recognition.close()

if args.video_total_save:
    print("closing video_writer_total")
    video_writer_total.close()
if args.video_original_save:
    print("closing video_writer_original")
    video_writer_original.close()    
if args.use_wandb:
    if args.video_total_save:
        print("Saving video_writer_total in wandb")
        artifact = wandb.Artifact("video_result", type="video")
        artifact.add_file(video_writer_total.filename)
        wandb.run.log_artifact(artifact)

        #Save video mp4
        input_avi_path = video_writer_total.filename
        output_mp4_path = video_writer_total.filename.replace(".avi",".mp4")
        vision.convert_avi_to_mp4(input_avi_path, output_mp4_path,fps=int(1.0/fps_total))
        caption = args.facedetection_model if args.facedetection_use else ""
        caption = (caption+"_"+args.facerecognition_model) if args.facerecognition_use else caption
        caption = (caption+"_"+args.facetracking_model) if args.facetracking_use else caption
        print("caption = ",caption)
        print("fps_total =",int(1.0/fps_total))
        print("Sending video to wandb..")
        wandb.log({"video_result_log": wandb.Video(output_mp4_path, fps=int(1.0/fps_total), format="mp4",caption=caption)})
    wandb.finish()