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
    return final_ids which is in pic and ann file
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

def one_label_COCO(image_folder:str, ann_file:str):
  """
    return image id and uniq label (Select the class with the largest number of objects)
    image_folder: path to the image folder
    ann_file: path to the annotation file
  """
  # retirn final_ids which is in pic and ann file
  image_ids = extract_ids(image_folder, ann_file)
  # Define coco
  coco = COCO(ann_file)   
  # Create a dictionary to store the final labels
  img_to_label = {}

  # Processing each image
  for img_id in image_ids:
      # Get image annotations
      ann_ids = coco.getAnnIds(imgIds=img_id)
      anns = coco.loadAnns(ann_ids)
      
      # Count the number of objects of each class
      class_counts = {}
      for ann in anns:
          class_id = ann['category_id']
          class_counts[class_id] = class_counts.get(class_id, 0) + 1
      
      # Select the class with the largest number of objects.
      if class_counts:
          dominant_class = max(class_counts, key=class_counts.get)
          img_to_label[img_id] = dominant_class
      else:
          img_to_label[img_id] = None  # If the image has no objects

  #Sort Dict by id
  img_to_label = dict(sorted(img_to_label.items()))

  return img_to_label



def pic_preproc_Intersection(image_folder:str, ann_file:str, size: int, test_ration:float):
  """
  resize and add 1 dimention to the image.  return just Intersection images. split train and test 
  image_folder: path to the image folder
  ann_file: path to the annotation file
  size: size of Length and width
  test_ration: test ration (float)
  """
  id_Intersection = extract_ids(image_folder, ann_file)
  # A list to save the paths of the desired images
  filtered_image_paths = []

  # Reading image paths
  image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(".jpg")]

  # Filter image paths based on IDs in id_Intersection
  for image_path in image_paths:
      image_name = os.path.basename(image_path)
      id_number = int(image_name.split('.')[0])  #Extracting ID from image name
      if id_number in id_Intersection:
          filtered_image_paths.append(image_path)  # Add image path to filtered list
  # Sort the path
  filtered_image_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
  print(filtered_image_paths)
  #####Preprocess function
  all_images = []
  # main function
  for img_path in filtered_image_paths:
      img = cv2.imread(img_path)
      gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      resized_img = cv2.resize(gray_img1, (size, size))
      gray_img2 = resized_img[:, :, np.newaxis]
      # Normalize pixel values to be between 0 and 1
      gray_img2 = gray_img2 / 255.0 

      # Add processed image to list
      all_images.append(gray_img2)
  # Convert list to NumPy array
  all_pic = np.array(all_images)
  # Split train and test
  train_pic, test_pic = train_test_split(pic, test_size=test_ration, random_state=42)

  return all_pic, train_pic, test_pic


def onehot_labels(my_dict:dict, num_classes:int):
  """
  changes the dictionary which has id and classes to one hot array
  my_dict: main dictionary
  num_classes: number of classes
  """
  # Number of classes
  num_classes = num_classes
  # Create a 2D array of the form (number of data, num_classes)
  num_data = len(my_dict)
  onehot_array = np.zeros((num_data, num_classes))
  # Converting a dictionary to a one-hot two-dimensional array
  for idx, (key, val) in enumerate(my_dict.items()):
      onehot_array[idx][val] = 1
  return onehot_array