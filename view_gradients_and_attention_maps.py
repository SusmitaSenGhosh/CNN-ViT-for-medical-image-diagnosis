#%% BN  #%%
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import tensorflow_addons as tfa
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import cv2
import numpy as np 
import random
import TransResNet,TransResNet_Aux, my_visualize, my_visualize_aux
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import suppli as spp
import os
from load_data import *
from load_model import *
from sklearn.utils import class_weight
tf.compat.v1.enable_eager_execution()

# %% set paths and parameters
data_name = 'chestxray2' #select one from the list ['cbis_ddsm_2class','cbis_ddsm_5class', 'DR','Col_Hist','ISIC18','chestxray1','chestxray2','BHI']
model_name = ['ResNet50','TransResNet','TransResNet_Aux']
batch_size = 16 
epochs = 1
input_size = 224
seed = 0
fileEnd ='.h5'
weight_path1 = 'C:/Users/susmi/OneDrive/Desktop/test/' +'/'+data_name+'/'+model_name[0]+'/model.h5' 
weight_path2 = 'C:/Users/susmi/OneDrive/Desktop/test/' +'/'+data_name+'/'+model_name[1]+'/model.h5' 
weight_path3 = 'C:/Users/susmi/OneDrive/Desktop/test/' +'/'+data_name+'/'+model_name[2]+'/model.h5' 
savePath = 'C:/Users/susmi/OneDrive/Desktop/test/' +'/'+data_name+'/'+'attention_maps/'

if not os.path.exists(savePath):
    os.makedirs(savePath)

def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   os.environ['TF_DETERMINISTIC_OPS'] = '1'
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   
   
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
no_class = len(np.unique(labelTr))

if data_name == 'cbis_ddsm_2class' or data_name == 'cbis_ddsm_5class' or data_name == 'DR' or data_name == 'BHI':
    scaling_factor = 255
else:
    scaling_factor = 1

if data_name == 'cbis_ddsm_2class' or data_name == 'cbis_ddsm_5class':
    alpha  = 0.25
else:
    alpha = 0.5
no_sample = 20

#%% load model for ResNet50
model = load_model(model_name[0], no_class,weights)
model.load_weights(weight_path1)

base_model = tf.keras.applications.ResNet50(include_top=True, input_tensor=None, input_shape=(224,224,3), pooling='avg')


x = model.layers[0].layers[-1].output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation = tfa.activations.gelu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(no_class, 'softmax')(x)
model1 = Model(inputs=model.layers[0].layers[0].input, outputs=x) # second input provided to model

for i in range(1,7):
    model1.layers[-i].set_weights(model.layers[-i].get_weights())

print(model.predict(np.expand_dims(trainS[10],axis = 0)))
print(model1.predict(np.expand_dims(trainS[10],axis = 0)))
del model
#%%

model = load_model(model_name[1], no_class,weights)
model.load_weights(weight_path2)

x = model.layers[0].layers[-1].output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation = tfa.activations.gelu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(no_class, 'softmax')(x)
model2 = Model(inputs=model.layers[0].layers[0].input, outputs=x) # second input provided to model

for i in range(1,7):
    model2.layers[-i].set_weights(model.layers[-i].get_weights())

print(model.predict(np.expand_dims(trainS[10],axis = 0)))
print(model2.predict(np.expand_dims(trainS[10],axis = 0)))
del model
#%%

model3 = load_model(model_name[2], no_class,weights)
model3.summary()
model3.load_weights(weight_path3)

print(model3.predict(np.expand_dims(trainS[10],axis = 0)))

#%%
def make_gradcam_heatmap2(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # if pred_index is None:
        #     pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    #heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = (heatmap-tf.math.reduce_min(heatmap))/ (tf.math.reduce_max(heatmap)-tf.math.reduce_min(heatmap))

    return heatmap.numpy()
def make_gradcam_heatmap3(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # if pred_index is None:
        #     pred_index = tf.argmax(preds[0])
        class_channel = preds[2][:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    #heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = (heatmap-tf.math.reduce_min(heatmap))/ (tf.math.reduce_max(heatmap)-tf.math.reduce_min(heatmap))

    return heatmap.numpy()
def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):


    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    jet_heatmap = jet_heatmap/jet_heatmap.max()
    #print(jet_heatmap.max(),jet_heatmap.min())

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap* alpha + img
    superimposed_img = superimposed_img/superimposed_img.max()
    #superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
   # superimposed_img.save(cam_path)

    # Display Grad CAM
    return jet_heatmap,superimposed_img


        #%%
for i in range(0,no_class):
    count1 = 0
    indices = np.where(labelTs==i)
    count2 = 0 
    while(True):
        index = indices[0][count1]
        img =testS[index]
        img_array = np.expand_dims(img,axis = 0)
        class_index = int(labelTs[index])
        
        predicted_class_index1 = np.argmax(model1.predict(img_array))
        predicted_class_index2 = np.argmax(model2.predict(img_array))
        predicted_class_index3 = np.argmax(model3.predict(img_array)[2])

        print(index,class_index,predicted_class_index1)
        print(index,class_index,predicted_class_index2)
        print(index,class_index,predicted_class_index3)
        if (predicted_class_index2 == class_index ):#& predicted_class_index2 == class_index & predicted_class_index3 == class_index):
            count2 = count2+1

            last_conv_layer_name =  "conv5_block3_3_conv"
            model1.layers[-1].activation = None
            preds = model1.predict(img_array)
            heatmap1 = make_gradcam_heatmap2(img_array, model1, last_conv_layer_name, pred_index = class_index)
            heatmap1, _ = save_and_display_gradcam(img, heatmap1, cam_path="cam.jpg", alpha=0.5)
            
            last_conv_layer_name = "conv5_block2_out"
            model2.layers[-1].activation = None
            preds = model2.predict(img_array)
            heatmap2_1 = make_gradcam_heatmap2(img_array, model2, last_conv_layer_name, pred_index = class_index)
            heatmap2_1, _ = save_and_display_gradcam(img, heatmap2_1, cam_path="cam.jpg", alpha=0.5)
            
            heatmap2_2,_ = my_visualize.attention_map(model2, img_array,alpha)        
            
            last_conv_layer_name = "conv5_block2_out"
            model3.layers[-1].activation = None
            preds = model3.predict(img_array)
            heatmap3_1 = make_gradcam_heatmap3(img_array, model3, last_conv_layer_name, pred_index = class_index)
            heatmap3_1, _ = save_and_display_gradcam(img, heatmap3_1, cam_path="cam.jpg", alpha=0.5)
            
            heatmap3_2,_ = my_visualize_aux.attention_map(model3, img_array,alpha)        
            
            
            img = img/scaling_factor
            superimposed1 = heatmap1*alpha+img
            superimposed1 = superimposed1/superimposed1.max()
            
            
            superimposed2_1 = heatmap2_1*alpha+img
            superimposed2_1 = superimposed2_1/superimposed2_1.max()
            
            superimposed2_2 = heatmap2_2*alpha+img
            superimposed2_2 = superimposed2_2/superimposed2_2.max()
        
            comb_heatmap2 = (heatmap2_1+heatmap2_2)/2 
            superimposedcomb2 = comb_heatmap2*alpha+img
            superimposedcomb2 = superimposedcomb2/superimposedcomb2.max()
            
            
            superimposed3_1 = heatmap3_1*alpha+img
            superimposed3_1 = superimposed3_1/superimposed3_1.max()
            
            superimposed3_2 = heatmap3_2*alpha+img
            superimposed3_2 = superimposed3_2/superimposed3_2.max()
        
            comb_heatmap3 = (heatmap3_1+heatmap3_2)/2 
            superimposedcomb3 = comb_heatmap3*alpha+img
            superimposedcomb3 = superimposedcomb3/superimposedcomb3.max()
            

            fig, axs = plt.subplots(1, 1)
            axs.imshow(img)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_image.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
        
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed1)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_superimposed1_'+str(class_index)+'_'+str(predicted_class_index1)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
        
        
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed2_1)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_superimposed2_1_'+str(class_index)+'_'+str(predicted_class_index2)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
            
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed2_2)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_superimposed2_2_'+str(class_index)+'_'+str(predicted_class_index2)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
            
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposedcomb2)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'superimposedcomb2_'+str(class_index)+'_'+str(predicted_class_index2)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
            
            
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed3_1)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_superimposed3_1_'+str(class_index)+'_'+str(predicted_class_index2)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
            
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed3_2)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_superimposed3_2_'+str(class_index)+'_'+str(predicted_class_index2)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
            
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposedcomb3)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'superimposedcomb3_'+str(class_index)+'_'+str(predicted_class_index2)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
        
        count1 = count1+1
        if count2 == no_sample or len(indices[0])==count2:
            break
