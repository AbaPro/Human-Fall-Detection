import cv2
import tensorflow as tf
import numpy as np
import math
import cv2

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    i = 0
    # Define variables for x and y coordinates of each keypoint
    nose_x, nose_y = 0, 0
    left_shoulder_x, left_shoulder_y = 0, 0
    left_hip_x, left_hip_y = 0, 0
    left_ankle_x, left_ankle_y = 0, 0
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            if i == 0:
                nose_x, nose_y = int(kx), int(ky)
                cv2.circle(frame, (nose_x, nose_y), 4, (255, 255, 0), -1)
            else:
                if i == 5:
                    left_shoulder_x, left_shoulder_y = int(kx), int(ky)
                elif i == 11:
                    left_hip_x, left_hip_y = int(kx), int(ky)
                elif i == 15:
                    left_ankle_x, left_ankle_y = int(kx), int(ky)
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

                len_factor = math.sqrt(((left_shoulder_y - left_hip_y)**2 + (left_shoulder_x - left_hip_x)**2 ))
                if (left_shoulder_y > left_ankle_y - len_factor 
                    and left_hip_y > left_ankle_y - (len_factor / 2) 
                    and left_shoulder_y > left_hip_y - (len_factor / 2)
                    and left_shoulder_y> 0 and left_ankle_y >0
                    and left_hip_y > 0):
                    cv2.rectangle(frame,(int(0), int(0)),(int(100), int(100)),color=(0, 0, 255),thickness=5,lineType=cv2.LINE_AA)
        i += 1

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

def gen_frames():  
    interpreter = tf.lite.Interpreter(model_path='tf_models\lite-model_movenet_singlepose_thunder_3.tflite')
    interpreter.allocate_tensors()
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Reshape image
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
            # img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
            input_image = tf.cast(img, dtype=tf.float32)
            
            # Setup input and output 
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Make predictions 
            interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            interpreter.invoke()
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
            
            # Rendering 
            draw_connections(frame, keypoints_with_scores, EDGES, 0.3)
            draw_keypoints(frame, keypoints_with_scores, 0.3)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
