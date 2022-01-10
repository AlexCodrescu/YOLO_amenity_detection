from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl
import pandas as pd
import imageio

image_path = 'D:/1000_data/data_resized'
aug_image_path = 'D:/1000_data/data_aug'
os.chdir(image_path)
df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

print('creating df')

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

print('created df')

uniqueValues = df['filename'].unique()

df2 = df[['xmin', 'ymin', 'xmax', 'ymax', 'class']]

print('augmenting')
for i in range (0, len(uniqueValues)):
  img = cv2.imread(image_path + '/' + uniqueValues[i])[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
  #bboxes = [df['xmin'] ,df['ymin'], df['xmax'], df['ymax'], df['class']]
  #inspect the bounding boxes
  idx = df[df['filename'] == uniqueValues[i]].index
  bboxes = np.array(df2[idx[0]:idx[len(idx) - 1] + 1].to_numpy())
  # print("-" * 10 , i , '-' * 10)
  # rnd = random.randint(1, 7)
  rnd = 1

  # if rnd == 0:
  #   plotted_img = draw_rect(img, bboxes)
  #     plt.imshow(plotted_img)
  #     plt.show()
  
  if rnd == 1:
    img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
    # plotted_img = draw_rect(img_, bboxes_)
    # plt.imshow(plotted_img)
    # plt.show()

  if rnd == 2:
    img_, bboxes_ = RandomScale(0.3, diff = True)(img.copy(), bboxes.copy())
    # plotted_img = draw_rect(img_, bboxes_)
    # plt.imshow(plotted_img)
    # plt.show()

  if rnd == 3:
    img_, bboxes_ = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
    # plotted_img = draw_rect(img_, bboxes_)
    # plt.imshow(plotted_img)
    # plt.show()
  
  if rnd == 4:
    img_, bboxes_ = RandomRotate(20)(img.copy(), bboxes.copy())
    # plotted_img = draw_rect(img_, bboxes_)
    # plt.imshow(plotted_img)
    # plt.show()
  
  if rnd == 5:
    img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
    # plotted_img = draw_rect(img_, bboxes_)
    # plt.imshow(plotted_img)
    # plt.show()

  if rnd == 6:
    img_, bboxes_ = RandomHSV(100, 100, 100)(img.copy(), bboxes.copy())
    # plotted_img = draw_rect(img_, bboxes_)
    # plt.imshow(plotted_img)
    # plt.show()

  if rnd == 7:
    try:
        seq = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomTranslate(), RandomRotate(10), RandomShear()])
        img_, bboxes_ = seq(img.copy(), bboxes.copy())
    except:
        img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())


    # plotted_img = draw_rect(img_, bboxes_)
    # plt.imshow(plotted_img)
    # plt.show()

  #adding 'aug_' to each image that has been augmented
  imageio.imwrite(aug_image_path + '/' + 'aug_' + uniqueValues[i], img_)



  filtered_class_data = pd.DataFrame()


  filtered_class_data['classNumber'] = ''
  filtered_class_data['center x'] = ''
  filtered_class_data['center y'] = ''
  filtered_class_data['width'] = ''
  filtered_class_data['height'] = ''

#convering back to yolo format and saving files
  for j in range(0, len(bboxes_)):

    xmin = bboxes_[j][0] 
    ymin = bboxes_[j][1]
    xmax = bboxes_[j][2]
    ymax = bboxes_[j][3]
    cls = bboxes_[j][4]
    width = df['width'][idx[0]]
    height = df['height'][idx[0]]

    ImageID = uniqueValues[i]
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

  save_path = aug_image_path + '/' 'aug_' + uniqueValues[i][:-4] + '.txt'
  print(filtered_class_data.head())
  filtered_class_data.to_csv(save_path, header=False, index=False, sep=' ')

  # print(bboxes_)
print('augmented')