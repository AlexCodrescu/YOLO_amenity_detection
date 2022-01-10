import cv2
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import pandas as pd
import os

def fix_yolo_bbs(img, txt):

  data = txt.readlines()
  dh, dw, _ = img.shape
  bounding_boxes = []
  labels = []

  for dt in data:
    
    # Split string to float
    c, x, y, w, h = map(float, dt.split(' '))

    # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
    l = ((x - w / 2) * dw)
    r = ((x + w / 2) * dw)
    t = ((y - h / 2) * dh)
    b = ((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    #df = df.append([[jpg_image_title, 416, 416, c, l, t, r, b]], ignore_index = True)
    labels.append(c)
    #bounding_boxes.append([l, r, t, b])
    bounding_boxes.append([l + 1, r - 1, t + 1, b - 1])
    
  bbs_array = []
  for i in range(0, len(bounding_boxes)):
    bbs_array.append(BoundingBox(x1=bounding_boxes[i][0], x2=bounding_boxes[i][1], y1=bounding_boxes[i][2], y2=bounding_boxes[i][3]))
  
  #finding the bounding boxes that are clipping out of image by 1 pixel and shifting them inwards by 1 pixel
  bbs = BoundingBoxesOnImage(bbs_array, shape=img.shape)
  bbs = bbs.clip_out_of_image()
  #print(bbs)
  #print(bbs.to_xyxy_array())
  cut_bbs = bbs.to_xyxy_array()
  for i in range(0, len(cut_bbs)):
      cut_bbs[i][0] = cut_bbs[i][0] + 1
      cut_bbs[i][1] = cut_bbs[i][1] - 1
      cut_bbs[i][2] = cut_bbs[i][2] + 1
      cut_bbs[i][3] = cut_bbs[i][3] - 1
  return labels, cut_bbs


def convert_back_to_yolo_format(img_path, annotations_path):
  annotations = open(annotations_path, 'r')
  img = cv2.imread(img_path)

  filtered_class_data = pd.DataFrame()
  labels, bboxes_ = fix_yolo_bbs(img, annotations)
  annotations.close()
  dh, dw, _ = img.shape


  filtered_class_data['classNumber'] = ''
  filtered_class_data['center x'] = ''
  filtered_class_data['center y'] = ''
  filtered_class_data['width'] = ''
  filtered_class_data['height'] = ''


  for j in range(0, len(bboxes_)):

    xmin = bboxes_[j][0] 
    ymin = bboxes_[j][1]
    xmax = bboxes_[j][2]
    ymax = bboxes_[j][3]
    cls = labels[j]
    width = dw
    height = dh

    classNumber = cls

    dwidth = 1./width
    dheight = 1./height
    center_x = (xmax + xmin) / 2.0
    center_y = (ymax + ymin) / 2.0
    width = xmax - xmin
    height = ymax - ymin

    center_x = center_x * dwidth
    width = width * dwidth
    center_y = center_y * dheight
    height = height * dheight

    filtered_class_data.loc[-1] = [classNumber, center_x, center_y, width, height]  # adding a row
    filtered_class_data.index = filtered_class_data.index + 1  # shifting index
    filtered_class_data = filtered_class_data.sort_index()

  annotations = open(annotations_path, 'w')
  for i in range(0, len(filtered_class_data)):
    annotations.write(str(int(filtered_class_data['classNumber'][i])) + ' ' + str(filtered_class_data['center x'][i]) + ' ' + str(filtered_class_data['center y'][i]) + ' ' + str(filtered_class_data['width'][i]) + ' ' + str(filtered_class_data['height'][i]) + '\n')    
  annotations.close()


image_path = 'D:/1000_data/aug_and_normal'

os.chdir(image_path)

#converting back to yolo format and saving
for curent_dir, dirs, files in os.walk('.'):
  for f in files:
    if f.endswith('.jpg'):
      image_title = f
      image_txt_title = f[:-4] + '.txt'
      convert_back_to_yolo_format(image_path + '/' + image_title, image_path + '/' + image_txt_title)

