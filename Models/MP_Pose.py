import mediapipe as mp
import threading
import queue

import cv2
import numpy as np

# LINE_BODY and COLORS_BODY are used when drawing the skeleton in 3D. 
rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0)}
LINES_BODY = [[12,6],[11,3],[3,6],[11,13],[13,15],[15,17],[18,16],[16,14],[14,12],[12,11],[11,23],[24,12],[24,23],[24,26],[26,28],[28,32],[27,31],[27,25],[23,25]]

COLORS_BODY = ["middle","right","left",
                "right","right","right","right","right",
                "middle","middle","middle","middle",
                "left","left","left","left","left",
                "right","right","right","left","left","left"]
COLORS_BODY = [rgb[x] for x in COLORS_BODY]

class Renderer:
    def __init__(self, width, height):
        self.frame = None
        self.pause = False
        self.image_width = width
        self.image_height = height
        self.blank = np.zeros((height,width,3), dtype=np.uint8)

        # Rendering flags
        self.show_rot_rect = False
        self.show_landmarks = True
        self.show_score = False
        self.show_fps = True
        self.reset = False

    def is_present(self, body, lm_id):
        return body[lm_id].visibility > 0.1

    def draw_landmarks(self, body):
        if self.show_landmarks:                
            list_connections = LINES_BODY
            lines = np.asarray([[[int(body[point].x*self.image_width), int(body[point].y*self.image_height)]for point in line] for line in list_connections if self.is_present(body, line[0]) and self.is_present(body, line[1])])
            cv2.polylines(self.frame, lines, False, (0, 255, 255), 5, cv2.LINE_AA) #rgb(241, 196, 15)

            for i,x_y in enumerate(body):
                cv2.circle(self.frame, (int(x_y.x*self.image_width), int(x_y.y*self.image_height)), 4, (255,255,0), -11) #rgb(243, 156, 18)
        
    def draw(self, body):
        self.frame = self.blank.copy()
        if body:
            self.draw_landmarks(body)
        self.body = body
        return self.frame

class MP_Pose:
    def __init__(self, height, width) -> None:
        #Setup renderer
        self.renderer = Renderer(width, height)

        #Setup pose estimation
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
        self.renderer.image_height=height
        self.renderer.image_width=width
        self.body = None

        #Queue to hold the most recent frame
        self.frame_queue = queue.Queue(maxsize=1)  

        #Create processing thread
        self.processing_thread = threading.Thread(target=self._process_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_thread(self):
        while True:
            # Wait for the most recent frame
            image = self.frame_queue.get()
            if image is None:
                break  # Stop the thread if a None image is passed
            # Process the frame
            self.body = self.process(image)

    def update_frame(self, frame):
        if not self.frame_queue.empty():
            self.frame_queue.get()  # Discard the previous frame if the queue is full
        self.frame_queue.put(frame)

    def stop_processing(self):
        self.frame_queue.put(None)  # Send a signal to stop the processing thread
        self.processing_thread.join()
            
    def process(self, image):
        result = self.pose.process(image)
        body = None
        if result.pose_landmarks != None:
            body = result.pose_landmarks.landmark
        return body
    
    def draw(self, frame, body):
        prediction = self.renderer.draw(body=body)
        frame_output = cv2.addWeighted(frame, 1, prediction, 1, 1)
        return frame_output, prediction[:,:,1]
    
    def valid(self):
        return self.body
    
    def check_overlap(self, output, current):
        #If the overlap between the prediction and background = 0 and the prediction overlap with foreground is > 1000 pixels
        return np.sum(np.logical_and(output, np.logical_not(current))) == 0 and np.sum(np.logical_and(output, current)) > 1000