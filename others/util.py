import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from google.colab.patches import cv2_imshow


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