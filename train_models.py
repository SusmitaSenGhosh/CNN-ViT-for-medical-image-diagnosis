#%% import packages #%%

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
#from tensorflow.keras.layers import Input
#from tensorflow.keras.models import Model
#import tensorflow_addons as tfa
import numpy as np
import random
import os
import suppli as spp
#import cv2
import time
from sklearn.utils import class_weight
from load_data import *
from load_model import *
import statistics as stat

tf.compat.v1.enable_eager_execution()

# %% set paths and parameters
data_name = 'chestxray2' #select one from the list ['cbis_ddsm_2class','cbis_ddsm_5class', 'DR','Col_Hist','ISIC18','chestxray1','chestxray2','BHI']
model_name = 'TransResNet_Aux'  #select one from the list ['ResNet50','TransResNet','TransResNet_Aux']
batch_size = 16 
epochs = 1
input_size = 224
seed = 0
fileEnd ='.h5'
weight_path = 'C:/Users/susmi/OneDrive/Desktop/test/' +'/'+data_name+'/'+model_name+'/' # path is address of the deatination folder

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   os.environ['TF_DETERMINISTIC_OPS'] = '1'
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)

def exp_decay(epoch):
    initial_lrate = 0.0002
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    print('lr:',lrate)
    return lrate

def evaluate_performance(conf): 
    n = conf.shape[0]
    acsa = 0
    acsp = 0
    acsf = 0
    acc_list = []
    for i in range(0,n):
        acsa = acsa + conf[i,i]/sum(conf[i,:])
        acc_list.append(round(100*conf[i,i]/sum(conf[i,:]),2))
        acsp = acsp + conf[i,i]/sum(conf[:,i])
        acsf = acsf +  2* (conf[i,i]/sum(conf[i,:]))*(conf[i,i]/sum(conf[:,i]))/((conf[i,i]/sum(conf[i,:]))+(conf[i,i]/sum(conf[:,i])))
    acsa = round(100*acsa/n,2)
    acsp = round(100*acsp/n,2)
    acsf = round(100*acsf/n,2)
    return acsa,acsp,acsf,round(np.sum(np.absolute(acc_list-stat.mean(acc_list)))/len(acc_list),2)
#%% load data
reset_random_seeds(0)
trainS, labelTr, testS, labelTs = load_data(data_name)
no_class = len(np.unique(labelTr))
labelsCat=to_categorical(labelTr)

shuffleIndex=np.random.choice(np.arange(labelTr.shape[0]), size=(labelTr.shape[0],), replace=False)
trainS=trainS[shuffleIndex]
labelTr=labelTr[shuffleIndex]
labelsCat=labelsCat[shuffleIndex]

weights = class_weight.compute_class_weight('balanced', np.unique(labelTr),labelTr)
classes = list(np.unique(labelTr))
class_weights = {classes[i]: weights[i] for i in range(len(classes))}

#%% load model
model = load_model(model_name, no_class,weights)
#%% train model

acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((epochs,),dtype = np.float32), np.zeros((epochs,),dtype = np.float32), np.zeros((epochs,),dtype = np.float32)
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((epochs,),dtype = np.float32), np.zeros((epochs,),dtype = np.float32), np.zeros((epochs,),dtype = np.float32)
confMatSaveTr,confMatSaveTs=np.zeros((epochs, no_class, no_class),dtype = np.float32), np.zeros((epochs, no_class, no_class),dtype = np.float32)
tprSaveTr, tprSaveTs=np.zeros((epochs, no_class),dtype = np.float32), np.zeros((epochs, no_class),dtype = np.float32)

epoch = 0
while epoch<epochs:
    start_time = time.time()

    K.set_value(model.optimizer.learning_rate, exp_decay(epoch))  # set new learning_rate
    if model_name == 'TransResNet_Aux':
        model.fit(trainS, [labelsCat,labelsCat,labelsCat],batch_size=batch_size, verbose=1)#, class_weight = class_weights)
       
        p = model.predict(trainS)
        pLabel=np.argmax(p[2], axis=1)
        acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
        print('Train: epoch: ', epoch, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
        print('TPR: ', np.round(tpr, 2))
        acsaSaveTr[epoch], gmSaveTr[epoch], accSaveTr[epoch]=acsa, gm, acc
        confMatSaveTr[epoch]=confMat
        tprSaveTr[epoch]=tpr

        p = model.predict(testS)
    
        pLabel=np.argmax(p[2], axis=1)
        acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
        print('Test: epoch: ', epoch, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
        print('TPR: ', np.round(tpr, 2))
        acsaSaveTs[epoch], gmSaveTs[epoch], accSaveTs[epoch]=acsa, gm, acc
        confMatSaveTs[epoch]=confMat
        tprSaveTs[epoch]=tpr
        
        
    else :
        model.fit(trainS, labelsCat,batch_size=batch_size, verbose=1, class_weight = class_weights)
    
        pLabel=np.argmax(model.predict(trainS), axis=1)
        acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
        print('Train: epoch: ', epoch, 'ACSA: ', np.round(acsa, 4))
        print('TPR: ', np.round(tpr, 2))
        acsaSaveTr[epoch], gmSaveTr[epoch], accSaveTr[epoch]=acsa, gm, acc
        confMatSaveTr[epoch]=confMat
        tprSaveTr[epoch]=tpr
            
        pLabel=np.argmax(model.predict(testS), axis=1)
        acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
        print('Test: epoch: ', epoch, 'ACSA: ', np.round(acsa, 4))
        print('TPR: ', np.round(tpr, 2))
        acsaSaveTs[epoch], gmSaveTs[epoch], accSaveTs[epoch]=acsa, gm, acc
        confMatSaveTs[epoch]=confMat
        tprSaveTs[epoch]=tpr
    
    epoch = epoch+1
    end_time = time.time() 
    print("total time taken ", epoch, " loop: ", end_time - start_time)

model.save(weight_path+'/model'+fileEnd)


recordSave=weight_path+'Record'
np.savez(recordSave, acsaSaveTr=acsaSaveTr, gmSaveTr=gmSaveTr, accSaveTr=accSaveTr,acsaSaveTs=acsaSaveTs, gmSaveTs=gmSaveTs, accSaveTs=accSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs, tprSaveTr=tprSaveTr, tprSaveTs=tprSaveTs)

acsa,acsp,acsf,mada = evaluate_performance(confMat)
print('acsa:', acsa,'acsp:', acsp,'acsf:',acsf,'mada:',mada )