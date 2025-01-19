import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def hsv(img, hue: int, saturation:int, value:int):
  """"
change the HSV value
img: orginal image
hue: hue chanel
saturation: hue saturation
value: hue value
  """""
  #change to HSV
  hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  saturated_image = cv2.merge([cv2.add(hsv_image[:, :, 0], hue), cv2.add(hsv_image[:, :, 1], saturation), cv2.add(hsv_image[:, :, 2], value)])
  #change to BGR
  saturated_image = cv2.cvtColor(saturated_image, cv2.COLOR_HSV2BGR)
  return saturated_image
  
 
 def brightness(img, operation: str, value: int):
  """"
add or subtract the brightness
img: orginal image
operation: have to choose "add" or "subtract"
value: value we want to add or subtract
  """""
  #value of add or subtract
  M = np.ones(img.shape, dtype = "uint8") * value
  #operation
  added = cv2.add(img, M)
  subtracted = cv2.subtract(img, M)
  #chose operation
  if operation == "add":
      imgg = cv2.add(img, M)  
  elif operation == "subtract":
      imgg = cv2.subtract(img, M)  
  else:
      print("chose operation")

  return imgg


def contrast(img, gamma: int):
  """"
increase or decrease contrast
img: orginal image
gamma: if gamma is bigger than 1 then contrast decreases And vice versa.
  """""
  lookUpTable = np.empty((1,256), np.uint8)
  for i in range(256):
      lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
  out = cv2.LUT(img, lookUpTable)
  return out
  
  
def blur(img, threshold: int, hight_th: int, low_th: int ):
  """"
blur filter
img: orginal image
threshold: Threshold for length and width of images (1000)
hight_th:Divide the length and width above the threshold by this number for the kernel size (70)
low_th:Divide the length and width below the threshold by this number for the kernel size (50)
  """""
  height, width = img.shape[:2]
  if height > threshold or width > threshold:
      kernel_size = (height // hight_th, width // hight_th)
  else:
      kernel_size = (height // low_th, width // low_th)

  gaussian1 = cv2.blur(img, kernel_size)
  return gaussian1

def rotate_and_scale_image(img, rotation_amount_degree, scale_factor):
  """"
rotate_and_scale_image
img: orginal image
rotation_amount_degree
scale_factor
  """""
  height, width, _ = img.shape
  theta = rotation_amount_degree * np.pi / 180.0

  # Define rotation matrix
  rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), rotation_amount_degree, scale_factor)

  # Calculate new dimensions after rotation
  cos_theta = np.abs(rotation_matrix[0, 0])
  sin_theta = np.abs(rotation_matrix[0, 1])
  new_width = int((height * sin_theta) + (width * cos_theta))
  new_height = int((height * cos_theta) + (width * sin_theta))

  # Adjust the rotation matrix to account for translation due to the change in size
  rotation_matrix[0, 2] += (new_width / 2) - (width / 2)
  rotation_matrix[1, 2] += (new_height / 2) - (height / 2)

  # Apply the rotation and scaling to the image
  result = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

  return result