import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
import sys

sys.path.append('/content/drive/MyDrive/Ai_Lab/cods/augment_simple')
from function2 import *


def filter1(img_load_path: str, img_save_path: str):
  """"
load, changen, and save img
img_load_path: path to the main img
img_save_path:path to save imgs
  """""
  # make directori for save path
  os.makedirs(img_save_path, exist_ok=True)
  # read imgs
  image_paths = [os.path.join(img_load_path, img) for img in os.listdir(img_load_path) if img.endswith(".JPG") or img.endswith(".jpg")]

  # main function
  for img_path in image_paths:
      img = cv2.imread(img_path)

      # function
      img_changed = filter_r1(img)

      # save changed img
      img_r_path = os.path.join(img_save_path, os.path.basename(img_path))
      cv2.imwrite(img_r_path, img_changed)
      
      
img_load_path = "/content/drive/MyDrive/Ai_Lab/data/yazdanian/"
img_save_path = "/content/drive/MyDrive/Ai_Lab/data/r_yazdanian3/"
filter1(img_load_path,img_save_path )