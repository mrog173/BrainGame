import numpy as np
import cv2
from datetime import datetime
import time
from utilities.Camera import Camera
from utilities.leaderboard_functions import *
from utilities.other import *
from utilities.game_state import GameState

def display_leaderboard(cap, renderer, leaderboard):
    #Display the leaderboard over the background frame from the camera
    image = cap.RGB_frame.get()
    renderer.update_frame(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Get output from the model
    output_frame, _ = renderer.draw(frame=np.zeros(image.shape, dtype=np.uint8), body=renderer.body)
    output_frame[:,:,0] = np.clip(output_frame[:,:,0], 80, 255)

    #Make it pretty
    output_frame = display_leaderboards(leaderboard, output_frame)
    image = cv2.addWeighted(output_frame,0.6,image,0.4,0)
    cv2.imshow("Video output", image)

    key = cv2.waitKey(1) 
    return key


def run_game(cap, renderer, gs):
    #This is the main game function
    frame = cap.RGB_frame.get()

    #Calculate FPS
    gs.frame_count += 1
    if gs.frame_count % 10 == 0:
        gs.end_time = time.time()
        #If we want to display FPS
        if gs.fps:
            gs.fps = 10 / (gs.end_time - gs.start_time)
        gs.start_time = gs.end_time

    #Update game screen
    if gs.change_bg and gs.times["EndScreen"] < datetime.now():
        gs.generateGameWall()

    renderer.update_frame(frame)

    output_frame, prediction = renderer.draw(frame=gs.current[1], body=renderer.body)

    #If time ran out
    if gs.times['GameTime'] < datetime.now():
        display_gameover(gs.score)
        return "Gameover"
    
    #If the result is correct: Flash green
    elif gs.change_bg and gs.isCorrect: 
        output_frame = cv2.addWeighted(output_frame,0.3,gs.correct_screen,0.7,0)

    #If the result is incorrect and time is up: Flash red
    elif gs.change_bg and not gs.isCorrect:
        output_frame = cv2.addWeighted(output_frame,0.3,gs.timerunout_screen,0.7,0)

    #If a prediction has been made
    elif renderer.valid(): #If landmarks are found
        if len(renderer.body) > 1:
            #Calculate correctness
            if renderer.check_overlap(prediction, gs.current[1][:,:,0]): #Check if pose is estimated as being correct, and trigger a background change
                output_frame = win_round(output_frame, gs)
            
            elif gs.times["BackgroundEnd"] < datetime.now(): #Check if time has run out and trigger a background change
                output_frame = lose_round(output_frame, gs)

            else: #Add blue tint
                output_frame[:,:,0] = np.clip(output_frame[:,:,0], 80, 255)

    #Check if time has run out and trigger a background change
    elif gs.times["BackgroundEnd"] < datetime.now(): 
        output_frame = lose_round(output_frame, gs)

    #Add blue tint
    else: 
        output_frame[:,:,0] = np.clip(output_frame[:,:,0], 80, 255)

    dst = cv2.addWeighted(output_frame,0.6,cv2.cvtColor(cap.RGB_frame.get(), cv2.COLOR_RGB2BGR),0.4,0)
    key = display_output(dst, gs)
    return key

def main(model_type, inputSrc, flip):
    game_mode = "Leaderboard"

    leaderboard = load_leaderboard()

    cv2.namedWindow("Video output", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    img = cv2.imread("LoadingScreen1.png")

    cv2.imshow("Video output", img)
    cv2.waitKey(1) #Wait for 1ms for some reason

    #Initialise game state
    gamestate = GameState(model_type)

    cap = Camera(gamestate.height, gamestate.width, inputSrc, flip)

    #Choose model
    if model_type == "Pose":
        from Models.MP_Pose import MP_Pose
        renderer = MP_Pose(gamestate.height, gamestate.width)
    elif model_type == "Segmentation":
        from Models.YOLO import YOLO
        renderer = YOLO(gamestate.height, gamestate.width)

    frame = cap.RGB_frame.get()
    renderer.update_frame(frame)

    #Main loop
    while cap.isOpened():
        if game_mode == "Timed" or game_mode == "Untimed":
            key = run_game(cap, renderer, gamestate)
        elif game_mode == "Leaderboard":
            key = display_leaderboard(cap, renderer, leaderboard)

        #Check if quit is triggered
        if key == 27 or key == ord('q'):
            break
        #If time has expired and the gameover flag is returned
        if key == "Gameover":
            update_leaderboard(leaderboard, model_type, gamestate.score)
            game_mode = "Leaderboard"
        #Toggle FPS
        elif key == ord('f'):
            if gamestate.fps:
                gamestate.fps = None
            else:
                gamestate.fps = 1
        #Toggle leaderboard
        elif key == ord('l'):
            game_mode = "Leaderboard"
            gamestate.reset()
        elif key == ord('s'):
            game_mode = "Timed"
            gamestate.reset()
            gamestate.show_time = True
        elif key == ord('u'):
            game_mode = "Untimed"
            gamestate.reset()
            gamestate.times['GameTime'] = datetime.now() + timedelta(days=1)
            gamestate.show_time = False

    cap.release()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the script with specified parameters.")
    parser.add_argument("--model_type", type=str, default="Segmentation", help="Type of model to use")
    parser.add_argument("--inputSrc", type=int, default=0, help="Input source for the model. Set this to 0 if only one webcam is connected or for the default webcam.")
    parser.add_argument("--flip", action="store_true", default=False, help="Whether to flip the input horizontally. Set to True if webcam is upside down.")

    args = parser.parse_args()

    if args.model_type not in ["Segmentation", "Pose"]:
        raise ValueError(f"Invalid model_type '{args.model_type}'. Please choose either 'Segmentation' or 'Pose'.")

    main(model_type=args.model_type, inputSrc=args.inputSrc, flip=args.flip)