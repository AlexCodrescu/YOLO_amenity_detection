import pandas as pd
import os

os.chdir('D:/oid_clean/OIDv4_ToolKit-master/OID/csv_folder')
classes_data = pd.read_csv('class-descriptions-boxable.csv', header=None)

classes = ['Table', 'Desk', 'Closet', 'Cupboard', 'Sink', 'Bathtub', 'Television', 'Refrigerator', 'Gas stove', 'Bed', 'Couch', 'Chair', 'Microwave oven', 'Shower', 'Pillow', 'Bathroom cabinet', 'Wardrobe', 'Bookcase', 'Coffee table', 'Coffeemaker']

#Check if specified classes are available in Open Images
class_strings = []
for j in classes:
  for i in range (0, len(classes_data)):
    if j == classes_data[1][i]:
      string = classes_data[0][i]
      print(string)
      class_strings.append(string)

#Create a dataframe with required columns
annotation_data = pd.read_csv('train-annotations-bbox.csv',
                              usecols=['ImageID',
                                       'LabelName',
                                       'XMin',
                                       'XMax',
                                       'YMin',
                                       'YMax'])

#filling dataframe with specified labels
filtered_class_data = annotation_data.loc[annotation_data['LabelName'].isin(class_strings)].copy()

filtered_class_data['classNumber'] = ''
filtered_class_data['center x'] = ''
filtered_class_data['center y'] = ''
filtered_class_data['width'] = ''
filtered_class_data['height'] = ''

#converting open images fromat to yolo format
for i in range(len(class_strings)):
  filtered_class_data.loc[filtered_class_data['LabelName'] == class_strings[i], 'classNumber'] = i

filtered_class_data['center x'] = (filtered_class_data['XMax'] + filtered_class_data['XMin']) / 2
filtered_class_data['center y'] = (filtered_class_data['YMax'] + filtered_class_data['YMin']) / 2

filtered_class_data['width'] = filtered_class_data['XMax'] - filtered_class_data['XMin']
filtered_class_data['height'] = filtered_class_data['YMax'] - filtered_class_data['YMin']

YOLO_values = filtered_class_data.loc[:, ['ImageID', 'classNumber', 'center x', 'center y', 'width', 'height']].copy()

image_path = 'D:/1000_data/data'
os.chdir(image_path)

#saving selected bounding boxes for each image
for curent_dir, dirs, files in os.walk('.'):
  for f in files:
    if f.endswith('.jpg'):
      image_title = f[:-4]
      YOLO_file = YOLO_values.loc[YOLO_values['ImageID'] == image_title]

      df = YOLO_file.loc[:, ['classNumber', 'center x', 'center y', 'width', 'height']].copy()

      save_path = image_path + '/' + image_title + '.txt'

      df.to_csv(save_path, header=False, index=False, sep=' ')