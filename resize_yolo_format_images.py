import os
import cv2
import pandas as pd
import imgaug as ia
ia.seed(1)
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
import xml.etree.ElementTree as ET
import shutil

def bbs_obj_to_df(bbs_object):
#     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
#     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs

# image_path = '/content/drive/MyDrive/augment_test'
image_path = 'D:/1000_data/data'
os.chdir(image_path)
df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
print('starting creation of df')
#converting yolo format to xmin ymin xmax ymax
for curent_dir, dirs, files in os.walk('.'):
  for f in files:
    if f.endswith('.jpg'):
      jpg_image_title = f
      txt_image_title = f[:-4] + '.txt'
      fl = open(image_path + '/' + txt_image_title, 'r')
      data = fl.readlines()
      img = cv2.imread(image_path + '/' + jpg_image_title)
      dh, dw, _ = img.shape
      fl.close()

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
        arr = [jpg_image_title, dw, dh, c, l, t, r, b]
        df = df.append(dict(zip(df.columns, arr)), ignore_index=True)

print('creation of df completed')

# height_resize = iaa.Sequential([ 
#     iaa.Resize({"height": 416, "width": 'keep-aspect-ratio'})
# ])
# width_resize = iaa.Sequential([ 
#     iaa.Resize({"height": 'keep-aspect-ratio', "width": 416})
# ])


height_resize = iaa.Sequential([ 
    iaa.Resize({"height": 416, "width": 416})
])
width_resize = iaa.Sequential([ 
    iaa.Resize({"height": 416, "width": 416})
])



def resize_imgaug(df, images_path, aug_images_path, image_prefix):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax']
                             )
    grouped = df.groupby('filename')    
    
    for filename in df['filename'].unique():
    #   Get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        
    #   The only difference between if and elif statements below is the use of height_resize and width_resize augmentors
    #   deffined previously.

    #   If image height is greater than or equal to image width 
    #   AND greater than 416px perform resizing augmentation shrinking image height to 416px.
        if group_df['height'].unique()[0] >= group_df['width'].unique()[0] and group_df['height'].unique()[0] > 416:
        #   read the image
            image = imageio.imread(images_path+filename)
        #   get bounding boxes coordinates and write into array        
            bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        #   pass the array of bounding boxes coordinates to the imgaug library
            bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        #   apply augmentation on image and on the bounding boxes
            image_aug, bbs_aug = height_resize(image=image, bounding_boxes=bbs)

            bbs_aug = bbs_aug.remove_out_of_image()
            bbs_aug = bbs_aug.clip_out_of_image()

        #   write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug)  
        #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)        
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
        #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
        #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
        #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
        #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])
            
    #   if image width is greater than image height 
    #   AND greater than 416px perform resizing augmentation shrinking image width to 416px
        elif group_df['width'].unique()[0] > group_df['height'].unique()[0] and group_df['width'].unique()[0] > 416:
        #   read the image
            image = imageio.imread(images_path+filename)
        #   get bounding boxes coordinates and write into array        
            bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        #   pass the array of bounding boxes coordinates to the imgaug library
            bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        #   apply augmentation on image and on the bounding boxes
            image_aug, bbs_aug = width_resize(image=image, bounding_boxes=bbs)

            bbs_aug = bbs_aug.remove_out_of_image()
            bbs_aug = bbs_aug.clip_out_of_image()

        #   write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug)  
        #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)        
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
        #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
        #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
        #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
        #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])

    #     append image info without any changes if it's height and width are both less than 600px 
        else:
            aug_bbs_xy = pd.concat([aug_bbs_xy, group_df])
    # return dataframe with updated images and bounding boxes annotations 
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy

print('resizing images')
resized_images_df = resize_imgaug(df, 'D:/1000_data/data/', 'D:/1000_data/data_resized/', '')
print('resized images')

#converting images back to yolo format and saving them
filtered_class_data = pd.DataFrame()
filtered_class_data['ImageID'] = resized_images_df['filename']
filtered_class_data['classNumber'] = resized_images_df['class']
filtered_class_data['center x'] = ''
filtered_class_data['center y'] = ''
filtered_class_data['width'] = ''
filtered_class_data['height'] = ''
filtered_class_data['dwidth'] = ''
filtered_class_data['dheight'] = ''

filtered_class_data['dwidth'] = 1./resized_images_df['width']
filtered_class_data['dheight'] = 1./resized_images_df['height']
filtered_class_data['center x'] = (resized_images_df['xmax'] + resized_images_df['xmin']) / 2.0
filtered_class_data['center y'] = (resized_images_df['ymax'] + resized_images_df['ymin']) / 2.0
filtered_class_data['width'] = resized_images_df['xmax'] - resized_images_df['xmin']
filtered_class_data['height'] = resized_images_df['ymax'] - resized_images_df['ymin']

filtered_class_data['center x'] = filtered_class_data['center x'] * filtered_class_data['dwidth']
filtered_class_data['width'] = filtered_class_data['width'] * filtered_class_data['dwidth']
filtered_class_data['center y'] = filtered_class_data['center y'] * filtered_class_data['dheight']
filtered_class_data['height'] = filtered_class_data['height'] * filtered_class_data['dheight']

YOLO_values = filtered_class_data.loc[:, ['ImageID', 'classNumber', 'center x', 'center y', 'width', 'height']].copy()

# image_path = '/content/drive/MyDrive/augment_test'
image_path = 'D:/1000_data/data_resized'
os.chdir(image_path)

print('creating bboxes')

for curent_dir, dirs, files in os.walk('.'):
  for f in files:
    if f.endswith('.jpg'):
      image_title = f[:-4]
      YOLO_file = YOLO_values.loc[YOLO_values['ImageID'] == f]

      df = YOLO_file.loc[:, ['classNumber', 'center x', 'center y', 'width', 'height']].copy()

      save_path = image_path + '/' + image_title + '.txt'

      df.to_csv(save_path, header=False, index=False, sep=' ')

print('created bboxes')