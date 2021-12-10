import cv2
import numpy as np
import tensorflow as tf

def load_checkpoint():
    """
    Loads the checkpoint file.
    """
    # Loads the detection graph from the model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(CHKPOINT_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess


def detect_hands(img, graph, session):
    """
    Detects hands in the image.
    Parameters:
        img: Image to detect hands in
    Returns:
        Detection boxes, detection scores, classes, and number of hands
    """
    # Input tensor is the image
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    # Output tensors are the detection boxes, scores, and classes
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    # Number of objects detected
    num_detections = graph.get_tensor_by_name('num_detections:0')
    # Actual detection
    (boxes, scores, classes, num) = session.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: np.expand_dims(img, axis=0)})

    return boxes, scores, classes, num


def collide_objects(num_hands, boxes, scores, objects, img):
    """
    Function that accounts for object collision, and returns a list of remaining objects
    Parameters:
        num_hands: Number of hands detected
        boxes: Detection boxes for the hand
        scores: Detection scores for the hand
        objects: List of tuples of coordinates and radius of objects (x, y, r)
        img: Image to detect hands in
    Returns:
        List of remaining objects
    """
    for i in range(num_hands):
        # Checks if the score is above the threshold
        if scores[i] > 0.5:
            # Gets the bounding box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            # Gets the bounding box dimensions
            h = ymax - ymin
            w = xmax - xmin
            # Gets the center of the bounding box
            center_x = xmin + (w / 2)
            center_y = ymin + (h / 2)
            # Checks if the center of the bounding box is within the radius of the object
            # If it is, then the object is removed
            for obj in objects:
                if (center_x - obj[0]) ** 2 + (center_y - obj[1]) ** 2 < obj[2] ** 2:
                    objects.remove(obj)
            
            # Draws a rectangle around the hand
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    return objects
