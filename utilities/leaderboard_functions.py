from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
import tkinter as tk
from tkinter import simpledialog

#Create a popup for gathering high scores
def get_player_name(category):
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Show the input dialog
    name = simpledialog.askstring(
        "New High Score!",
        f"🎉 Congratulations! You've achieved a new high score in {category}! 🏆\n\nPlease enter your name to record your achievement:"
    )
    root.destroy()
    return name

# Load leaderboard from JSON file
def load_leaderboard(filename="leaderboard.json"):
    with open(filename, "r") as file:
        leaderboard = json.load(file)
    return leaderboard

# Update the leaderboard file with text input
def update_leaderboard(leaderboard, category, score, filename="leaderboard.json"):
    top_scores = leaderboard[category]
    if len(top_scores) < 5 or score > min(top_scores, key=lambda x: x['score'])['score']:
        name = get_player_name(category)
        # Add the new score
        top_scores.append({"name": name, "score": score})
        # Sort the leaderboard by score in descending order
        top_scores.sort(key=lambda x: x['score'], reverse=True)
        # Keep the top five
        leaderboard[category] = top_scores[:5]

        with open(filename, "w") as file:
            json.dump(leaderboard, file, indent=4)

# Produce an image with the leaderboard image
def display_leaderboards(leaderboard, img):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Pixellettersfull-BnJ5.ttf", 80)

    # Position settings
    pose_start_y = 60
    segmentation_start_y = 580
    line_height = 80

    # Leaderboard
    draw.text((1000, 0), "Leaderboard", fill="white", font=font)

    # Display the Pose leaderboard
    draw.text((50, pose_start_y - 60), "Pose Leaderboard", fill="white", font=font)
    for idx, entry in enumerate(leaderboard["Pose"]):
        draw.text((50, pose_start_y + idx * line_height), f"{idx + 1}. {entry['name']}: {entry['score']}", fill="white", font=font)

    # Display the Segmentation leaderboard
    draw.text((50, segmentation_start_y - 60), "Segmentation Leaderboard", fill="white", font=font)
    for idx, entry in enumerate(leaderboard["Segmentation"]):
        draw.text((50, segmentation_start_y + idx * line_height), f"{idx + 1}. {entry['name']}: {entry['score']}", fill="white", font=font)

    return np.array(img)