from pathlib import Path
import sys, os

#Fix path so we can use YOLO
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # YOLO root directory
if str(os.path.join(str(ROOT), "yolov9")) not in sys.path:
    sys.path.append(os.path.join(str(ROOT), "yolov9"))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from yolov9.utils.general import Profile
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.general import non_max_suppression, check_img_size
from yolov9.utils.segment.general import process_mask
import threading
import queue
import cv2
import numpy as np

class YOLO:
    def __init__(self, height, width) -> None:
        #Setup model on GPU
        device = torch.device("cuda:0")

        self.imgsz = (640,640)
        self.height = height
        self.width = width

        #Setup model
        self.model = DetectMultiBackend("yolov9-c-seg.pt", device=device, dnn=False, data=ROOT, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else 1, 3, *self.imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        self.body = None

        self.imgsz = check_img_size(self.imgsz, s=self.stride)

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

            new_frame = cv2.resize(image, (640,640))
            new_frame = new_frame.transpose(2,0,1)
            
            # Process the frame
            self.body = self.process(new_frame)

    def update_frame(self, frame):
        if not self.frame_queue.empty():
            self.frame_queue.get()  # Discard the previous frame if the queue is full
        self.frame_queue.put(frame)

    def stop_processing(self):
        self.frame_queue.put(None)  # Send a signal to stop the processing thread
        self.processing_thread.join()
            
    def process(self, image):
        with self.dt[0]:
            im = torch.from_numpy(image).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        with self.dt[1]:
            pred, proto = self.model(im, augment=False, visualize=False)[:2]

        with self.dt[2]:
            pred = non_max_suppression(pred, 0.25, 0.45, 0, False, max_det=1000, nm=32)
        
        masks_temp = np.zeros((self.height,self.width),dtype=bool)
        for i, det in enumerate(pred):
            if len(det):
                masks = process_mask(proto[2].squeeze(0), det[:, 6:], det[:, :4], (640, 640), upsample=True)  # HWC
                masks = masks[0].cpu().detach().numpy()
                masks = cv2.resize(masks, (self.width,self.height), interpolation=cv2.INTER_NEAREST)
                masks_temp = np.logical_or(masks_temp,masks)
        masks_temp = 255*masks_temp.astype(np.uint8)
        
        return masks_temp
    
    def draw(self, frame, body):
        if isinstance(body, np.ndarray):
            prediction = np.stack((body,)*3,axis=-1).astype(np.uint8)
            prediction[:,:,0] = 0
            frame_output = cv2.addWeighted(prediction,1,frame,1,1)
            return frame_output, prediction[:,:,1]
        else:
            frame_output = frame.copy()
            prediction = np.zeros((self.height,self.width), dtype=np.uint8)
            return frame_output, prediction
    
    def valid(self):
        return isinstance(self.body, np.ndarray)
    
    def check_overlap(self, output, current):
        #If the overlap between the prediction and background = 0 and the prediction overlap with foreground is > 1000 pixels
        return np.sum(np.logical_and(output, np.logical_not(current))) == 0 and np.sum(output) > 10000 and np.sum(np.logical_and(output, current)) > 10000