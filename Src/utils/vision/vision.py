import cv2
import numpy as np
from .camera import *
from .fps_writer import *
from .video_writer import *
from .video_converter import *

def put_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    img = cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    img = cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return img

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=2,
          font_thickness=1,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    img = cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    img = cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return img, text_w, text_h

def draw_boundingboxes(image,predictions,color=(255,0,0),thickness=2,if_wh=False,is_track=False):
    if is_track:
        for pre in predictions:
            arreglo = np.array(pre, dtype=int)
            arreglo[arreglo < 0]=0
            [x1, y1, x2, y2, id] = arreglo 
            image = cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness=thickness)
            image, w, h = draw_text(image,str(id), pos=(x1, y1-10),font_scale=1,text_color_bg=color)
        return image
   
    if if_wh and len(predictions)>0:
        for [xmin,ymin,width,height] in predictions:
            start_point = (int(xmin),int(ymin))
            end_point   = (int(xmin+width),int(ymin+height))
            image = cv2.rectangle(image, start_point, end_point, color, thickness=thickness)
        
        return image
    
    if not if_wh  and len(predictions)>0:
        for [xmin,ymin,xmax,ymax] in predictions:
            start_point = (int(xmin),int(ymin))
            end_point   = (int(xmax),int(ymax))
            image = cv2.rectangle(image, start_point, end_point, color, thickness=thickness)
        
    return image




def draw_boxes_names(image,predictions,names,colors,thickness=2,font_scale=1):
    for name,pre,color in zip(names,predictions,colors):
        arreglo = np.array(pre, dtype=int)
        arreglo[arreglo < 0]=0
        [x1, y1, x2, y2, id] = arreglo 
        image = cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness=thickness)
        image, w, h = draw_text(image,name, pos=(x1, y1-10),font_scale=font_scale,text_color_bg=color)

    return image