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