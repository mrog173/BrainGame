import numpy as np
import os
import cv2
import random
from datetime import datetime, timedelta

class GameState:
    #Use to keep track of all the game related variables
    def __init__(self,model_type)->None:
        self.times = {
            "BackgroundEnd": 0,
            "EndScreen": 0,
            "GameTime": 0
        }

        self.change_bg = False
        self.isCorrect = False
        self.score = 0
        self.backgrounds = []
        self.show_time = True

        #For keeping track of FPS
        self.fps = None
        self.frame_count = 0

        self.width = 1920
        self.height = 1080

        #Define times
        self.wait_length = timedelta(seconds=1)
        self.round_length = timedelta(seconds=8)
        self.game_length = timedelta(seconds=20)
        self.start_time = 0
        self.end_time = 0

        #Define overlay screens
        correct_color = (113, 204, 46)
        error_color = (60, 76, 231)

        #Screen overlays
        self.correct_screen = np.full((self.height, self.width, 3), correct_color, dtype=np.uint8)
        self.timerunout_screen = np.full((self.height, self.width, 3), error_color, dtype=np.uint8)

        self.loadBackgrounds()

        overlay = cv2.imread('Overlay.png', cv2.IMREAD_UNCHANGED)
        self.overlay_rgb = overlay[:, :, :3]  # The RGB channels
        overlay_alpha = overlay[:, :, 3] / 255.0  # The alpha channel (normalized to 0-1 range)
        self.overlay_alpha = np.stack([overlay_alpha]*3, axis=-1)

        self.current = random.choice(self.backgrounds) #Set initial background

    def loadBackgrounds(self):
        #Load possible game backgrounds
        
        maps = os.listdir("GameMaps/")
        for i, pth in enumerate(maps):
            im = cv2.imread("GameMaps/"+pth)
            #Threshold backgrounds to prevent feathered edges
            im[im<120] = 0
            im[im>=120] = 255
            self.backgrounds += [(i, im)]

        print("Loaded", len(self.backgrounds), "backgrounds")

    def reset(self):
        self.times['BackgroundEnd'] = datetime.now() + self.round_length
        self.times['GameTime'] = datetime.now() + self.game_length
        self.change_bg = False
        self.isCorrect = False
        self.score = 0

    def generateGameWall(self):  
        #Ensure the same background isn't selected twice
        i = self.current[0]
        while i == self.current[0]:
            i, new_bg = random.choice(self.backgrounds)
        self.current = (i, new_bg)

        self.times["EndScreen"] = datetime.now() + self.wait_length
        self.times["BackgroundEnd"] = datetime.now() + self.round_length
        self.change_bg = False