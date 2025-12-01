from norfair import Tracker as Tracker_norfair, Detection
import numpy as np

class Tracker:
  def __init__(self,distance_function="euclidean", distance_threshold=100,hit_counter_max=3,period=2):
    try:
      self.tracker = Tracker_norfair(distance_function=distance_function,
                            distance_threshold=distance_threshold,hit_counter_max=hit_counter_max)
    except:
      self.tracker = Tracker_norfair(distance_function="mean_euclidean",
                            distance_threshold=distance_threshold,hit_counter_max=hit_counter_max)
    self.period = period
    
  def predict(self, locations):
      
    boxes_tracker = self.tracker.update(detections=[Detection(p) for p in np.array(locations)],period=self.period)
    # initializing_id, global_id
    det = []
    for box in boxes_tracker:
        det.append([box.estimate[0][0],box.estimate[0][1],box.estimate[0][2],box.estimate[0][3],box.id])
        #det.append(np.append(box.estimate[0],box.id ))
    return det
   