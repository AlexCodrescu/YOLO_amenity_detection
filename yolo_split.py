import os
import imageio

image_path = 'D:/1000_data/data'
save_folder = 'D:/1000_data/data_split/'
# image_path = '/content/drive/MyDrive/YOLO_V4/data/Dataset/train/images'
os.chdir(image_path)

#splitting the whole dataset in 20 subsets, each containing only one class for training an assembly of yolo nets on each class
for curent_dir, dirs, files in os.walk('.'):
  for f in files:

    arr_0 = []
    arr_1 = []
    arr_2 = []
    arr_3 = []
    arr_4 = []
    arr_5 = []
    arr_6 = []
    arr_7 = []
    arr_8 = []
    arr_9 = []
    arr_10 = []
    arr_11 = []
    arr_12 = []
    arr_13 = []
    arr_14 = []
    arr_15 = []
    arr_16 = []
    arr_17 = []
    arr_18 = []
    arr_19 = []

    if f.endswith('.jpg'):
      jpg_image_title = f
      txt_image_title = f[:-4] + '.txt'
      fl = open(image_path + '/' + txt_image_title, 'r')
      data = fl.readlines()
      img = imageio.imread(image_path + '/' + jpg_image_title)
      fl.close()

      for i in range(0, len(data)):
        cls = int(data[i].split()[0])
        if cls == 0:
          arr_0.append('0' + data[i][1:-1])
        elif cls == 1:
          arr_1.append('0' + data[i][1:-1])
        elif cls == 2:
          arr_2.append('0' + data[i][1:-1])
        elif cls == 3:
          arr_3.append('0' + data[i][1:-1])
        elif cls == 4:
          arr_4.append('0' + data[i][1:-1])
        elif cls == 5:
          arr_5.append('0' + data[i][1:-1])
        elif cls == 6:
          arr_6.append('0' + data[i][1:-1])
        elif cls == 7:
          arr_7.append('0' + data[i][1:-1])
        elif cls == 8:
          arr_8.append('0' + data[i][1:-1])
        elif cls == 9:
          arr_9.append('0' + data[i][1:-1])
        elif cls == 10:
          arr_10.append('0' + data[i][2:-1])
        elif cls == 11:
          arr_11.append('0' + data[i][2:-1])
        elif cls == 12:
          arr_12.append('0' + data[i][2:-1])
        elif cls == 13:
          arr_13.append('0' + data[i][2:-1])
        elif cls == 14:
          arr_14.append('0' + data[i][2:-1])
        elif cls == 15:
          arr_15.append('0' + data[i][2:-1])
        elif cls == 16:
          arr_16.append('0' + data[i][2:-1])
        elif cls == 17:
          arr_17.append('0' + data[i][2:-1])
        elif cls == 18:
          arr_18.append('0' + data[i][2:-1])
        elif cls == 19:
          arr_19.append('0' + data[i][2:-1])
        
      if len(arr_0) > 0:
        with open(save_folder + '0' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_0:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '0' + '/' + jpg_image_title, img)  

      if len(arr_1) > 0:
        with open(save_folder + '1' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_1:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '1' + '/' + jpg_image_title, img)  

      if len(arr_2) > 0:
        with open(save_folder + '2' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_2:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '2' + '/' + jpg_image_title, img) 

      if len(arr_3) > 0:
        with open(save_folder + '3' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_3:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '3' + '/' + jpg_image_title, img) 

      if len(arr_4) > 0:
        with open(save_folder + '4' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_4:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '4' + '/' + jpg_image_title, img) 

      if len(arr_5) > 0:
        with open(save_folder + '5' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_5:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '5' + '/' + jpg_image_title, img) 

      if len(arr_6) > 0:
        with open(save_folder + '6' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_6:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '6' + '/' + jpg_image_title, img) 

      if len(arr_7) > 0:
        with open(save_folder + '7' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_7:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '7' + '/' + jpg_image_title, img) 

      if len(arr_8) > 0:
        with open(save_folder + '8' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_8:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '8' + '/' + jpg_image_title, img) 

      if len(arr_9) > 0:
        with open(save_folder + '9' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_9:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '9' + '/' + jpg_image_title, img) 

      if len(arr_10) > 0:
        with open(save_folder + '10' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_10:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '10' + '/' + jpg_image_title, img) 

      if len(arr_11) > 0:
        with open(save_folder + '11' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_11:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '11' + '/' + jpg_image_title, img) 

      if len(arr_12) > 0:
        with open(save_folder + '12' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_12:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '12' + '/' + jpg_image_title, img) 

      if len(arr_13) > 0:
        with open(save_folder + '13' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_13:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '13' + '/' + jpg_image_title, img) 

      if len(arr_14) > 0:
        with open(save_folder + '14' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_14:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '14' + '/' + jpg_image_title, img) 

      if len(arr_15) > 0:
        with open(save_folder + '15' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_15:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '15' + '/' + jpg_image_title, img) 

      if len(arr_16) > 0:
        with open(save_folder + '16' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_16:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '16' + '/' + jpg_image_title, img) 

      if len(arr_17) > 0:
        with open(save_folder + '17' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_17:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '17' + '/' + jpg_image_title, img) 

      if len(arr_18) > 0:
        with open(save_folder + '18' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_18:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '18' + '/' + jpg_image_title, img) 

      if len(arr_19) > 0:
        with open(save_folder + '19' + '/' + txt_image_title, 'w') as filehandle:
          for listitem in arr_19:
            filehandle.write('%s\n' % listitem)
        imageio.imwrite(save_folder + '19' + '/' + jpg_image_title, img) 