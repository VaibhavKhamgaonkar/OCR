# =============================================================================
# Main file for Training
# =============================================================================

import keras
from keras.utils import multi_gpu_model
from keras.layers import Dense, Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization,MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop,Adam
from keras.models import Sequential,save_model,load_model
from keras import regularizers
#from keras.activations impport relu
import pandas as pd,numpy as np
import cv2
#from sklearn.metrics import classification_report,confusion_matrix
#import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.layers import Input, concatenate
from keras.models import Model
from keras.utils import plot_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#from AutoEncoder import StackedAutoEncoder
#from Functions import Functions
#from CNNTraining import CNNFunctions
#from OnlyDNN import Functions
from LSTMTraining import CNNFunctions
#creating an Object of the Class

#from keras import backend as K
import tensorflow as tf
# =============================================================================
# with K.tf.device('/gpu:1'):
#     config = tf.ConfigProto(intra_op_parallelism_threads=4,\
#            inter_op_parallelism_threads=4, allow_soft_placement=True,\
#            device_count = {'CPU' : 1, 'GPU' : 1})
#     session = tf.Session(config=config)
#     print('Done with Sess ........!')
#     K.set_session(session)
# =============================================================================
# =============================================================================
# with K.tf.device('/gpu:1'):
#     print('GPU Accessed')
#     config = tf.ConfigProto(intra_op_parallelism_threads=4,\
#            inter_op_parallelism_threads=4, allow_soft_placement=True,\
#            device_count = {'CPU' : 1, 'GPU' : 1})
#     session = tf.Session(config=config)
# K.set_session(session)
# 
# =============================================================================
print('Session Created....!!!')


path = os.path.dirname(os.path.abspath('__file__')) + '/'

obj = CNNFunctions(TrainingDataPath=path+'DATA_ver1/Train', valDataPath = path+'DATA_ver1/Validation',
                numClassses=26,lr=0.001, epochs=20, l2_labmda=0.0001, batchSize=256)

# Preparing the Model for training
#model,trainSamples,valSamples = obj.getModel()
#obj.getAutoEncoder()
#with tf.device('/gpu:1'):
obj.getModel()
    #Training the Model
obj.training()

# =============================================================================
# 
# stepsPerEpochs = trainSamples//32
# validationSteps = valSamples//32
# model.fit_generator(obj.genTrainData(),
#                          steps_per_epoch =stepsPerEpochs,
#                          epochs=50,
#                          validation_data = obj.genValidationData(),
#                          validation_steps = validationSteps,
#                          verbose = 1,
#                          #callbacks = [self.reduceLR,self.EarlyCheckPt,self.ModelCkPt]
#                          )
# =============================================================================
# =============================================================================
# 
# #loading model 
# model = load_model('/Users/vk250027/Documents/OCR work/Models/Version2_Specialchar91.2%/OCRModel_v1.h5')
# 
# def getData(img):
#     l=[]
#     img=cv2.resize(img,(32,32),cv2.INTER_AREA)
#     imgMatrix =img #np.array(img, dtype = 'uint8')
#     #print(imgMatrix.shape)
#     blurr = cv2.GaussianBlur(imgMatrix,(3,3), 0)
#     ret, thresh = cv2.threshold(blurr.copy(),0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY )
#     _,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if len(contours)==1:
#         cnt = contours[0]
#         area = cv2.contourArea(cnt)
#         perimeter = cv2.arcLength(cnt,True)
#         x,y,w,h = cv2.boundingRect(cnt)
#         aspect_ratio = float(w)/h
#         hull = cv2.convexHull(cnt)
#         hull_area = cv2.contourArea(hull)
#         try:
#             solidity = float(area)/hull_area
#         except ZeroDivisionError:
#             solidity= 0
#             
#         epsilon = 0.1*cv2.arcLength(cnt,True)
#         approx = cv2.approxPolyDP(cnt,epsilon,True)
#     
#     else:
#         #print('Found Multiple contours for the Image : ', len(contours))
#         for cnt in contours:
#         # =============================================================================
#         # Identifying the Feature from image
#         # =============================================================================
#             try:
#                 area = cv2.contourArea(cnt)
#                 perimeter = cv2.arcLength(cnt,True)
#                 x,y,w,h = cv2.boundingRect(cnt)
#                 aspect_ratio = float(w)/h
#                 hull = cv2.convexHull(cnt)
#                 hull_area = cv2.contourArea(hull)
#                 try:
#                     solidity = float(area)/hull_area
#                 except ZeroDivisionError:
#                     solidity= 0
#                     
#                 epsilon = 0.1*cv2.arcLength(cnt,True)
#                 approx = cv2.approxPolyDP(cnt,epsilon,True)
#                 break
#             except:
#                 print('Contour 1 has null enties hence moving for next contour value...')
#                 continue
#         
#     l.append([area, perimeter,aspect_ratio,solidity,len(approx)])
#     img = img.reshape(1,32,32,1)
#     return img,l
# 
# 
# 
# img = cv2.imread('/Users/vk250027/Documents/OCR work/ValidationSet/@/__0_38222.png',0)
# 
# 
# x,l = getData(img)
#     
#     
# 
# 
# model.predict( [x,np.array(l)])
# =============================================================================

# =============================================================================
# 
# 
# p = os.path.dirname(os.path.abspath('__file__')) + '/' 
# 
# p = p.split('/')
# p = p[:-2]
# p = '/'.join(p) + '/'
# 
# 
# p + 'main/xyz.py'
# 
# p + 'app/function.py'
# 
# 
# =============================================================================
