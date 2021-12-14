import cv2
import numpy as np
import tensorflow as tf

CHKPOINT_PATH = 'C:\\Users\\Kmilo\\Documents\\cs1430\\cv_mercury_final_project\\code\\handtracking\\hand_inference\\frozen_inference_graph.pb'

def load_checkpoint():
    """
    Loads the checkpoint file.
    """
    # Loads the detection graph from the model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(CHKPOINT_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
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

    return np.squeeze(boxes), np.squeeze(scores)


def collide_objects(num_hands, boxes, scores, objects, img, width, height):
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
    #for i in range(num_hands):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(1):
        # Checks if the score is above the threshold
        if scores[i] > 0.4:
            # Gets the bounding box coordinates
            (left, right, top, bottom) = (boxes[i][1] * width, boxes[i][3] * width,
                                          boxes[i][0] * height, boxes[i][2] * height)
            # Draws a rectangle around the hand
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            # Gets the center of the bounding box
            center_x = (left + right) // 2
            center_y = (top + bottom ) // 2 
            str_HandPos = ""+str(int(left))+","+ str(int(top))+" "+str(int(right))+","+ str(int(bottom))
            str_center_Hand = ""+ str(format(center_x, '.2f')) + "," + str(format(center_y,'.2f'))
            cv2.circle(img, (int(center_x),int(center_y)), 5, (0,255,0), 2)
            cv2.putText(img, 
                str_HandPos, 
                (950, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
            cv2.putText(img, 
                str_center_Hand, 
                (950, 80), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
          
            for obj in objects:
                 left, top, rigth, bott = obj.get_rectangle()
                 if center_x > left and center_x < rigth and center_y > top and center_y < bott:
                     objects.remove(obj)

    return objects
