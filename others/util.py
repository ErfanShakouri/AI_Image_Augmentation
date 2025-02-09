import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re

from google.colab.patches import cv2_imshow
from pycocotools.coco import COCO


def segment_draw(img_path:str, segment_point):
  '''

draw the segment on the image
img_path: path to the image
segment_point: list of points to draw the segment [[]]
  '''
  img = cv2.imread( img_path)
  # segment Coordinates
  segmentation = segment_point
  points = np.array(segmentation, np.int32).reshape((-1, 1, 2))
  #Draw a line around the area.
  cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
  return img


def bbox_draw(img_path:str, bbox_point):
  '''
draw the bbox on the image
img_path: path to the image
segment_point: list of points to draw the bbox []
  '''
  img = cv2.imread( img_path)
  # Coordinates
  bbox =bbox_point
  # Coordinates of the corners of the rectangle
  x, y, w, h = bbox
  x1, y1 = int(x), int(y)
  x2, y2 = int(x+w), int(y+h)
  # Draw a rectangle on the image
  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
  return img
  

def pic_preproc(img_load_path: str, size: int):
  """"
  resize and add 1 dimention to the image.  return all images
  img_load_path: path to the main img
  size: size of Length and width
  """""
  # read imgs
  image_paths = [os.path.join(img_load_path, img) for img in os.listdir(img_load_path) if img.endswith(".JPG") or img.endswith(".jpg")]
  # Create an empty list to store processed images
  all_images = []
  # main function
  for img_path in image_paths:
      img = cv2.imread(img_path)
      gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      resized_img = cv2.resize(gray_img1, (size, size))
      gray_img2 = resized_img[:, :, np.newaxis]
      # Normalize pixel values to be between 0 and 1
      gray_img2 = gray_img2 / 255.0 

      # Add processed image to list
      all_images.append(gray_img2)
  # Convert list to NumPy array
  x = np.array(all_images)
  return x
  

def extract_ids(image_folder:str, ann_file:str):
  """
    retirn final_ids which is in pic and ann file
    image_folder: path to the image folder
    ann_file: path to the annotation file
  """
  coco = COCO(ann_file)
  # List of ids in ann
  img_ids = coco.getImgIds()

  # path of images
  image_list = os.listdir(image_folder)
  id_list = []
  for image_name in image_list:
    if image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
      # Extracting ID using regular expression
      match = re.search(r'\d+', image_name)
      if match:
        image_id = int(match.group())
        id_list.append(image_id)

  # Intersection of two list
  final_id = list(set(img_ids) & set(id_list))

  return final_id