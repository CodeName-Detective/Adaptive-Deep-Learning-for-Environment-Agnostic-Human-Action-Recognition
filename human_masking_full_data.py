import numpy as np
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import pandas as pd
import mediapipe as mp
import os

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection = 1)

# Getting the actions for in the data set
# actions = os.listdir("Data/UCF-101")
actions = ['BalanceBeam', 'TennisSwing', 'Bowling', 'GolfSwing', 'Fencing', 'Biking', 'Diving', 'JavelinThrow', 'Archery', 'BasketballDunk']

# Creating a folder for masked background data
masked_data_folder = os.path.join("Data", "Human_Mask_Full_data")
os.makedirs(masked_data_folder, exist_ok=True)

# Creating folders for each action in the masked background data
for action in actions:
  action_folder = os.path.join("Data/Human_Mask_Full_data", action)
  os.makedirs(action_folder, exist_ok=True)

for bg_action in actions:
  videos = os.listdir(f"Data/UCF-101/{bg_action}")
  videos_present = os.listdir(f"Data/Human_Mask_Full_data/{bg_action}")
  videos = list(set(videos) - set(videos_present))
  for video in videos:
    segmentor = SelfiSegmentation()
    video_path = f'Data/UCF-101/{bg_action}/{video}'
    capture = cv2.VideoCapture(video_path)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output_path = f'Data/Human_Mask_Full_data/{bg_action}/{video}'
    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, size)
  
    while True:
      bg = cv2.imread('black.jpeg')
      bg = cv2.resize(bg, (320, 240))
      success, img = capture.read()
      if success is not True:
        break
      imgOut = segmentor.removeBG(img, bg)
      fg = cv2.copyTo(img,imgOut)
      msk = img - fg
      video_writer.write(msk)
    

# https://stackoverflow.com/questions/44073462/unable-to-save-background-subtracted-video-python-opencv
# https://answers.opencv.org/question/206471/how-do-i-save-a-video-with-background-subtraction-applied/
