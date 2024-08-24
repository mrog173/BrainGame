import cv2
from datetime import datetime, timedelta
from PIL import ImageFont, ImageDraw, Image
import numpy as np

def display_output(dst, gamestate, delay=1):
    #Add overlay text to the game screen
    dst = np.where(gamestate.overlay_alpha > 0, gamestate.overlay_rgb[:, :, :], dst[:, :, :])

    img_PIL = Image.fromarray(dst)
    draw = ImageDraw.Draw(img_PIL)
    font=ImageFont.truetype("Pixellettersfull-BnJ5.ttf", 80)

    #Display score, FPS, and time
    draw.text((30,10), f"SCORE: {gamestate.score:d}", font=font, fill=(0,0,0))
    if gamestate.fps:
        draw.text((1500,10), f'FPS: {gamestate.fps:.2f}', font=font, fill=(0,0,0))
    if gamestate.show_time:
        time_remaining = (gamestate.times['GameTime'] - datetime.now()).total_seconds()
        draw.text((770,10), f"Time: {time_remaining:.2f}", font=font, fill=(0,0,0))

    #Display image and get key result
    dst = np.array(img_PIL)
    cv2.imshow("Video output", dst)
    key = cv2.waitKey(delay) 
    return key

def display_gameover(score):
    #Display the game over screen
    img = cv2.imread("GameOver.png")
    img_PIL = Image.fromarray(img)
    draw = ImageDraw.Draw(img_PIL)
    font=ImageFont.truetype("Pixellettersfull-BnJ5.ttf", 160)
    draw.text((30,10), f"FINAL SCORE: {score:d}", font=font, fill=(255,255,255))
    font=ImageFont.truetype("Pixellettersfull-BnJ5.ttf", 80)
    draw.text((30,800), f"Press Esc to continue...", font=font, fill=(255,255,255))
    dst = np.array(img_PIL)
    cv2.imshow("Video output", dst)
    cv2.waitKey(0)

#Other functions...
def lose_round(output_frame, game_state):
    #End the round when the background time expires
    output_frame = cv2.addWeighted(output_frame,0.3,game_state.timerunout_screen,0.7,0)
    game_state.isCorrect = False
    if not game_state.change_bg:
        game_state.times["EndScreen"] = datetime.now() + timedelta(seconds=1)
        game_state.change_bg = True
    return output_frame

def win_round(output_frame, game_state):
    #End the round when the shape is made
    game_state.change_bg = True
    game_state.times["EndScreen"] = datetime.now() + timedelta(seconds=1)
    game_state.score += 1
    game_state.isCorrect = True
    #Flash green
    output_frame = cv2.addWeighted(output_frame,0.3,game_state.correct_screen,0.7,0)
    return output_frame