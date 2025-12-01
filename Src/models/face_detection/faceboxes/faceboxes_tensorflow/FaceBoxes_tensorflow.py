#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import os

class FaceDetector:
    def __init__(self, model_name="FaceBoxesProd.pb", gpu_memory_fraction=0.25, visible_device_list='0',device='gpu'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        local_path = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(local_path,'..','weights',model_name)
        
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.input_image = graph.get_tensor_by_name('import/image_tensor:0')
        self.output_ops = [
            graph.get_tensor_by_name('import/boxes:0'),
            graph.get_tensor_by_name('import/scores:0'),
            graph.get_tensor_by_name('import/num_boxes:0'),
        ]

        if device == 'gpu':
            gpu_options = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction,
                visible_device_list=visible_device_list
            )
            config_proto = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
            #config_proto.gpu_options.allow_growth = True

        else:
            config_proto=tf.ConfigProto(log_device_placement=True,device_count={'CPU':4})
        self.sess = tf.compat.v1.Session(graph=graph, config=config_proto)

    def close(self):
        self.sess.close()
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        del self.sess

    def predict(self, image, score_threshold=0.35):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            scores: a float numpy array of shape [num_faces].

        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """
        #image.flags.writeable = False

        h, w, _ = image.shape
        image = np.expand_dims(image, 0)

        boxes, scores, num_boxes = self.sess.run(
            self.output_ops, feed_dict={self.input_image: image}
        )
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        scaler = np.array([h, w, h, w], dtype='float32')
        boxes = boxes * scaler

        boxes_processed = []
        for box in boxes:
            boxes_processed.append([box[1],box[0],box[3],box[2]])
        #image.flags.writeable = True
        return boxes_processed, scores
