import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(data_name):
    if data_name == 'cbis_ddsm_2class':
        images=[]
        labels=[]
        feature_dictionary = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_normal': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
            }

        def _parse_function(example, feature_dictionary=feature_dictionary):
            parsed_example = tf.io.parse_example(example, feature_dictionary)
            return parsed_example

        def read_data(filename):
            full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE)
            # full_dataset = full_dataset.shuffle(buffer_size=31000)
            full_dataset = full_dataset.cache()
            print("Size of Training Dataset: ", len(list(full_dataset)))

            feature_dictionary = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_normal': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
            }   

            full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            for image_features in full_dataset:
                image = image_features['image'].numpy()
                image = tf.io.decode_raw(image_features['image'], tf.uint8)
                image = tf.reshape(image, [299, 299])        
                image=image.numpy()
                image=cv2.resize(image,(224,224))
                image=cv2.merge([image,image,image])        
                #plt.imshow(image)
                images.append(image)
                labels.append(image_features['label_normal'].numpy())


        filenames=['D:/Simpi/data/IDC/training10_0/training10_0.tfrecords',
                   # data/IDC/training10_1/training10_1.tfrecords',
                   #data/IDC/training10_2/training10_2.tfrecords',
                   # data/IDC/training10_3/training10_3.tfrecords',
                   # data/IDC/training10_4/training10_4.tfrecords']
                   ]


        for file in filenames:
            read_data(file)

        data = np.stack(images, axis=0 )
        label = np.asarray(labels)
        del images, labels

        trainS, testS, labelTr, labelTs = train_test_split(data, label, test_size=0.2, random_state=0,shuffle=False)
        del data, label

    elif data_name == 'cbis_ddsm_5class':
        images=[]
        labels=[]
        feature_dictionary = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_normal': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
            }

        def _parse_function(example, feature_dictionary=feature_dictionary):
            parsed_example = tf.io.parse_example(example, feature_dictionary)
            return parsed_example

        def read_data(filename):
            full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE)
            # full_dataset = full_dataset.shuffle(buffer_size=31000)
            full_dataset = full_dataset.cache()
            print("Size of Training Dataset: ", len(list(full_dataset)))
            
            feature_dictionary = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_normal': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
            }   

            full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            for image_features in full_dataset:
                image = image_features['image'].numpy()
                image = tf.io.decode_raw(image_features['image'], tf.uint8)
                image = tf.reshape(image, [299, 299])        
                image=image.numpy()
                image=cv2.resize(image,(224,224))
                image=cv2.merge([image,image,image])        
                #plt.imshow(image)
                images.append(image)
                labels.append(image_features['label'].numpy())

        filenames=['D:/Simpi/data/IDC/training10_0/training10_0.tfrecords',
                  # 'D:/Simpi/data/IDC/training10_1/training10_1.tfrecords',
                  # 'D:/Simpi/data/IDC/training10_2/training10_2.tfrecords',
                  # 'D:/Simpi/data/IDC/training10_3/training10_3.tfrecords',
                  # 'D:/Simpi/data/IDC/training10_4/training10_4.tfrecords']
                  ]
            
        for file in filenames:
            read_data(file)

        
        data = np.stack(images, axis=0 )
        label = np.asarray(labels)
        del images, labels

        trainS, testS, labelTr, labelTs = train_test_split(data, label, test_size=0.2, random_state=0,shuffle=True)
        del data, label
    elif data_name == 'DR':
        path = 'D:/data/processed/'
        temp = np.load(path+'/DR/sample.npz')
        data = temp['arr_0']
        label = np.asarray(temp['arr_1'])
        for n, i in enumerate(label):
            if i > 0:
                label[n] = 1 
        del temp

        trainS, testS, labelTr, labelTs = train_test_split(data, label, test_size=0.2, random_state=0,shuffle=True)
        del data, label
        
    elif data_name == 'Col_Hist':
        path = 'D:/data/processed/'
        temp = np.load(path+'/col_hist/sample_normalized.npz')
        data = temp['arr_0']
        #label = np.asarray(temp['arr_1'])
        label = temp['arr_1'] 
        del temp

        trainS, testS, labelTr, labelTs = train_test_split(data, label, test_size=0.2, random_state=0,shuffle=True)
        del data, label
    elif data_name == 'ISIC18':
        path = 'D:/data/processed/'
        temp = np.load(path+'/isic18/sample_normalized.npz')
        data = temp['arr_0']
        #label = np.asarray(temp['arr_1'],dtype = np.uint8)
        label = temp['arr_1']
        del temp

        trainS, testS, labelTr, labelTs = train_test_split(data, label, test_size=0.2, random_state=0,shuffle=True)
        del data, label
    
    elif data_name == 'chestxray1':
        def get_data_label(data_dir,text_file): 
            x = []
            y = []
            weights = []
            mapping={
                'normal': 0,
                'pneumonia': 1,
                'COVID-19': 2
            }
            line_content= text_file
            for i in range(0,len(line_content)):
                file_name = line_content[i].split(" ")[1].rstrip()
                x.append(cv2.imread(data_dir + file_name))
                y.append(line_content[i].split(" ")[2].rstrip())
            y =list(map(mapping.get, y))
            for j in range(0,len(list(set(y)))):
                weights.append(y.count(list(set(y))[j])) 
            weights = [element / sum(weights) for element in weights]
            return np.asarray(x,dtype = np.float32)/255, to_categorical(np.asarray(y,dtype = np.float32)), weights

        train_txt = 'D:/data/raw/chestxray1/train_txt_v2.txt'
        val_txt = 'D:/data/raw/chestxray1/val_txt_v2.txt'
        test_txt = 'D:/data/raw/chestxray1/test_COVIDx4.txt'
        preprocessed_image_path = 'D:/data/processed/chestxray1/'

        with open(test_txt, 'r') as fr:
             test_files = fr.readlines()
        with open(val_txt, 'r') as fr:
             val_files = fr.readlines()
        with open(train_txt, 'r') as fr:
             train_files = fr.readlines()

        train_files  = train_files + val_files
        trainS, y_train, _ = get_data_label(preprocessed_image_path,train_files)
        labelTr = np.argmax(y_train,axis = 1)
        del y_train
        testS, y_test, _ = get_data_label(preprocessed_image_path,test_files)
        labelTs = np.argmax(y_test,axis = 1)
        del y_test


    elif data_name == 'chestxray2':
        path = 'D:/data/processed/'
        temp = np.load(path+'/chestxary2/train.npz')
        trainS = temp['arr_0']
        labelTr = np.asarray(temp['arr_1'],dtype = np.uint8)
        del temp

        temp = np.load(path+'/chestxray2/test.npz')
        testS = temp['arr_0']
        labelTs = np.asarray(temp['arr_1'],dtype = np.uint8)
        del temp
        
    elif data_name == 'BHI':
        path = 'D:/data/processed/'
        temp = np.load(path+'/BHI/sample224.npz')
        data = temp['arr_0']
        #label = np.asarray(temp['arr_1'])
        label = temp['arr_1'] 
        del temp

        trainS, testS, labelTr, labelTs = train_test_split(data, label, test_size=0.2, random_state=0,shuffle=True)
        del data, label
    return trainS, labelTr, testS,labelTs