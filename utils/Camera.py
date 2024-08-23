import cv2
import threading
import queue

class Camera:
    def __init__(self, height, width, source, flip) -> None:
        self.cap = cv2.VideoCapture()
        self.cap.open(source + cv2.CAP_DSHOW)
        #print(cap.get(cv2.CAP_PROP_BUFFERSIZE), cap.get(cv2.CAP_PROP_FOURCC), cap.get(cv2.CAP_PROP_BACKEND))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.flip = flip

        self.processing_thread = threading.Thread(target=self._read_thread)
        self.processing_thread.daemon = True  # Make thread a daemon so it exits when the main program does
        self.processing_thread.start()

        self.capture = True
        #self.BGR_frame = queue.Queue(maxsize=1)
        self.RGB_frame = queue.Queue(maxsize=1)

    def isOpened(self):
        return self.cap.isOpened()

    def _read_thread(self):
        while self.capture:
            _, frame = self.cap.read()
            if self.flip:
                frame = cv2.flip(frame, 0)
            else:
                frame = cv2.flip(frame, 1)

            # self.BGR_frame.put(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            self.RGB_frame.put(frame)