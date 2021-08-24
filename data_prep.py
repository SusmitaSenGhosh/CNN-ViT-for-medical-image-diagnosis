import os
import numpy as np
import csv
import cv2
from PIL import Image 
import random

save_path = 'C:/Users/susmi/OneDrive/Desktop/test/data/'
#%% ISIC18
path = 'D:/Simpi/data/retinopathy/isic18/ISIC2018_Task3_Training_Input'
csv_path = 'D:/Simpi/data/retinopathy/isic18/ISIC_grounfthruth.csv'

classes = [0,1,2,3,4]
file_name = []
y = []
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(line_count)
        if line_count !=0:
            file_name.append(row[0])
            img = np.asarray(cv2.resize(cv2.imread(os.path.join(path,row[0]+'.jpg')),(224,224))/255, dtype =np.float32)
            y.append(np.argmax((row[1::])))
            if line_count ==1:
                x=np.expand_dims(img,axis = 0)
            else:
                x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        line_count += 1
        
np.savez(save_path+'/isic18/sample_normalized.npz', x, y,file_name)

#%% colorectal hist

path = 'D:/Simpi/data/retinopathy/Kather_texture_2016_image_tiles_5000'
x = []
y = []
dirs = os.listdir(path)
count = 0
class_name = 0
for folder in dirs:
    print(count)
    file_list = os.listdir(os.path.join(path,folder))
    for file in file_list:
        img = Image.open(os.path.join(path,folder,file))
        img = np.asarray(img.resize((224,224)),dtype = np.float32)/255
        if count == 0:
            x=np.expand_dims(img,axis = 0)
            y.append(class_name)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
            y.append(class_name)
        count = count+1
    class_name = class_name+1


np.savez(save_path+'/col_hist/sample_normalized.npz', x, y)
        
#%% diabetic retinopathy

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    

def preprocess_image(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = np.asarray(cv2.resize(image, (224, 224)),dtype = np.float32)
   # image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image


path = 'D:/Simpi/data/retinopathy/kaggle_diabetic_ratinopathy/train/'
classes = [0,1,2,3,4]
file_name = []
class_name = []
with open('D:/Simpi/data/retinopathy/kaggle_diabetic_ratinopathy/trainLabels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count !=0:
            file_name.append(row[0])
            class_name.append(int(row[1]))
        line_count += 1
        
indices_list = []
for c in classes:
    indices = [i for i, x in enumerate(class_name) if x == c]
    if c == 0:
        indices = random.sample(indices,7000)
    indices_list.extend(indices)
    
count = 0
y = []
for i in indices_list:
    print(count)
    img = preprocess_image(os.path.join(path,file_name[i]+'.jpeg'))
    if count == 0:
        x=np.expand_dims(img,axis = 0)
        y.append(class_name[i])
    else:
        x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        y.append(class_name[i])
    count = count +1


np.savez(save_path+'/kaggle_diabetic_ratinopathy/sample_no_prepro.npz', x, y,file_name,file_name,indices_list)

#%%    chestray2

classes = [0,1]
y = []
count = 0
class_count = 0
path = 'D:/Simpi/data/retinopathy/chest_xray/train'
folders = ['NORMAl','PNEUMONIA']
for folder in folders:
    file_name = os.listdir(os.path.join(path,folder))
    for file in file_name:
        img = np.asarray(cv2.resize(cv2.imread(os.path.join(path,folder,file)),(224,224))/255,dtype = np.float32)
        y.append(classes[class_count])
        if count ==0:
            x=np.expand_dims(img,axis = 0)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        count += 1
        print(count)
    class_count += 1
np.savez(save_path+'/chest_xray/train.npz', x, y,file_name)


classes = [0,1]
y = []
count = 0
class_count = 0
path = 'D:/Simpi/data/retinopathy/chest_xray/test'
folders = ['NORMAl','PNEUMONIA']
for folder in folders:
    file_name = os.listdir(os.path.join(path,folder))
    for file in file_name:
        img = np.asarray(cv2.resize(cv2.imread(os.path.join(path,folder,file)),(224,224))/255,dtype = np.float32)
        y.append(classes[class_count])
        if count ==0:
            x=np.expand_dims(img,axis = 0)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        count += 1
        print(count)
    class_count += 1
np.savez(save_path+'/chest_xray/test.npz', x, y,file_name)
#%% BHI


def get_data(data_folder_path,classes, folders):
    random.seed(0)
    count = 0
    skipped_count = 0
    y = []
    folders_count = 0
    
    for folder in folders:
        print(folders_count)
        for labels in classes:
            files_list = os.listdir(os.path.join(data_folder_path,folder,labels))
            for files in files_list:
                img = cv2.imread(os.path.join(data_folder_path,folder,labels,files))
                if count == 0:
                    if img.shape == (50,50,3):
                        x=np.expand_dims(cv2.resize(img,(224,224)),axis = 0)
                        y.append(int(labels))
                    else:
                        skipped_count = skipped_count+1
                        print('skipped : ',labels,skipped_count)
                        
                else:
                    if img.shape == (50,50,3):
                        x = np.concatenate((x,np.expand_dims(cv2.resize(img,(224,224)),axis = 0)),axis = 0)
                        y.append(int(labels))
                    else:
                        skipped_count = skipped_count+1
                        print('skipped : ',labels,skipped_count )
                count = count +1
        folders_count = folders_count+1

    return x,y


random.seed(0)
data_folder_path = 'C:/Users/susmi/data/IDC'
no_folders = 15
classes = ['0','1']
all_folders = os.listdir(data_folder_path)
folders = random.sample(all_folders,no_folders)
data, labels = get_data(data_folder_path,classes, folders)
np.savez(save_path+'/IDC_train_test/sample224.npz', data, labels)
#%%