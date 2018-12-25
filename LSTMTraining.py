import matplotlib
matplotlib.use("Agg")
import keras
#from keras.utils import multi_gpu_model
from keras.layers import Dense, Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization,MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop,Adam,SGD
from keras.models import Sequential,save_model,load_model
from keras import regularizers
from keras.layers import AvgPool2D
#from keras.activations impport relu
import pandas as pd,numpy as np
import cv2
#from sklearn.metrics import classification_report,confusion_matrix
#import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LSTM, Reshape, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
from keras.layers import Input, concatenate
from keras.models import Model
from keras.utils import plot_model
import os
import pickle

import matplotlib.pyplot as plt

from keras import backend as K
#import tensorflow as tf
with K.tf.device('/gpu:0'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,\
           inter_op_parallelism_threads=4, allow_soft_placement=True,\
           device_count = {'CPU' : 1, 'GPU' : 1})
    session = tf.Session(config=config)
    K.set_session(session)


#from keras import backend as K
#from alt_model_checkpoint import AltModelCheckpoint
#K.clear_session()

class CNNFunctions():
    #Intiailise the object with Data
    def __init__(self, TrainingDataPath, valDataPath, numClassses, lr = 0.001, 
                 epochs=50, l2_labmda = 0.0001, batchSize = 64):
        self.batchSize = batchSize
        self.path = os.path.dirname(os.path.abspath('__file__')) + '/'
        #Path of Training Data
        self.trainPath = TrainingDataPath
        #Path of Validation Data
        self.valDataPath = valDataPath
        #creating base model for multiGPU 
        self.baseModel = None
        #defining the ImageGenerator Object
        self.trainDataGen = ImageDataGenerator(
                                   #horizontal_flip = True
                                   )
        #self.testDataGen = ImageDataGenerator()
        self.valDataGen = ImageDataGenerator()
        
        self.reduceLR= ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                         mode='auto',min_delta=1e-4, cooldown=0, min_lr=0)

        self.EarlyCheckPt = EarlyStopping(monitor='val_loss', min_delta=0, 
                                          patience=5, verbose=1, mode='auto')

        
        self.ModelCkPt = ModelCheckpoint(self.path +'Models/Seperate_Models/'+ 'Alphabets_v3.h5', monitor='val_loss', 
                                         verbose=1, save_best_only=True, save_weights_only=False, 
                                         mode='auto', period=1)
        
        self.train_generator = self.trainDataGen.flow_from_directory(self.trainPath + '/',
                                          target_size = (32,32),
                                          class_mode = 'categorical',
                                          batch_size = self.batchSize,
                                          color_mode='grayscale',
                                          shuffle=True,
                                          seed=16)
        self.validation_generator = self.valDataGen.flow_from_directory(self.valDataPath + '/',
                                          target_size = (32,32),
                                          class_mode = 'categorical',
                                          batch_size = self.batchSize,
                                          color_mode='grayscale',
                                          shuffle=True,
                                          seed=16)
        
        self.LR = lr,
        self.l2Lambda = l2_labmda
        self.epochs = epochs
        self.optimizer = RMSprop(self.LR,decay=1e-5)
        self.numCategories = numClassses
        self.model = None
        self.hist = None
        with open(self.path +'Models/Seperate_Models/' + 'AlphabetsLabels_v3.pkl','wb') as f:
            pickle.dump(self.train_generator.class_indices,f)
        # =============================================================================
        #  Getting the Model       
        # =============================================================================
    def getModel(self):

        optim = self.optimizer
        input1 = Input(shape=(32,32,1),name='1stInput' )
        
        conv1 = Conv2D(24, (3,3),activation = 'relu', kernel_initializer='glorot_uniform',
                                     kernel_regularizer=regularizers.l2(self.l2Lambda),
                                     data_format='channels_last',name='1st')(input1)
        
        batchNorm1 = BatchNormalization(name='batchNorm1')(conv1)
        maxpool3 = MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same',name = 'pool3')(batchNorm1)
        drop3 = Dropout(0.25,name='dropout3')(maxpool3)
       
        conv2 = Conv2D(16, (3,3),activation = 'relu', kernel_initializer='glorot_uniform',
                                     kernel_regularizer=regularizers.l2(self.l2Lambda),
                                     data_format='channels_last',name='2nd_conv')(drop3)
        
        batchNorm2 = BatchNormalization(name='batchNorm2')(conv2)
        maxpool4 = MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same',name = 'pool4')(batchNorm2)
        drop4 = Dropout(0.25,name='dropout4')(maxpool4)
       
        
        flatForLSTM = TimeDistributed(Flatten(name='flatLayer_for_LSTM'))(drop4)
        lstm0 = LSTM(25,activation='tanh',return_sequences=True,dropout=0.25,
                     recurrent_dropout=0.25,recurrent_initializer='glorot_uniform',kernel_regularizer = regularizers.l2(self.l2Lambda),
                     recurrent_regularizer = regularizers.l2(self.l2Lambda),name='lstm_0')(flatForLSTM)
        
# =============================================================================
#         d1 = Dense(512, activation = 'sigmoid',kernel_initializer='he_uniform',
#                                     kernel_regularizer=regularizers.l2(self.l2Lambda), name='denseInput_1')(lstm0)
# 
#         
#         drop1 = Dropout(0.25,name='drpout_11')(d1)
#         d2 = Dense(512, activation = 'sigmoid',kernel_initializer='he_uniform',
#                                     kernel_regularizer=regularizers.l2(self.l2Lambda), name='denseInput_2')(drop1)
# 
#         
#         drop2 = Dropout(0.25,name='drpout2')(d2)
# =============================================================================

# =============================================================================
#         self.model=Sequential()
#         self.model.add(TimeDistributed(Conv2D(24, (3,3),activation = 'relu', kernel_initializer='he_uniform',
#                                      kernel_regularizer=regularizers.l2(self.l2Lambda),
#                                      data_format='channels_last',name='1st', input_shape = (32,32,1))))
#         
#         self.model.add(TimeDistributed(BatchNormalization(name='batchNorm1')))
#         self.model.add(TimeDistributed(Conv2D(16, (3,3),activation = 'relu',kernel_initializer='he_uniform',
#                                      kernel_regularizer=regularizers.l2(self.l2Lambda),
#                                     data_format='channels_last',name='convlayer2')))
#         self.model.add(TimeDistributed(BatchNormalization(name='batchNorm2')))
#         
#         self.model.add(TimeDistributed(Conv2D(10, (3,3), activation = 'relu',kernel_initializer='he_uniform',
#                                      kernel_regularizer=regularizers.l2(self.l2Lambda),
#                                      data_format='channels_last',name='convlayer3')))
#         
#         self.model.add(TimeDistributed(BatchNormalization(name='batchNorm3')))
#         
#         self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same',name = 'pool3')))
#         self.model.add(TimeDistributed(Dropout(0.25,name='dropout3')))
#         self.model.add(TimeDistributed(Flatten(name='flat0')))
#         self.model.add(LSTM(10,activation='tanh',return_sequences=True, dropout=0.25, name='lstm_1'))
#         
#         self.model.add(Dense(1024,activation='relu',kernel_initializer='glorot_uniform',
#                                     kernel_regularizer=regularizers.l2(self.l2Lambda), name='FUll_1') )
#         
#         self.model.add(Dropout(0.25,name='dropout4'))
#         
#         self.model.add(Dense(1024,activation='relu',kernel_initializer='glorot_uniform',
#                                     kernel_regularizer=regularizers.l2(self.l2Lambda), name='FUll_2'))
#         self.model.add(Dropout(0.25,name='dropout5'))
#     
#         self.model.add( Dense(self.numCategories, activation='softmax', name='FUll_3') )
#         
#         self.model.compile(optimizer=optim,loss='categorical_crossentropy',metrics=['accuracy'])
# =============================================================================
        #input1 = Input(shape=(32,32,1),name='1stInput' )
        
        #1st convolution Layer
# =============================================================================
#         conv1 = Conv2D(24, (3,3),activation = 'relu', kernel_initializer='he_uniform',
#                                     kernel_regularizer=regularizers.l2(self.l2Lambda),
#                                     data_format='channels_last',name='1st')(lstm0)
#                                 
#             
#         batchNorm1 = BatchNormalization(name='batchNorm1')(conv1)
#         
#         #2nd convolution Layer
#         conv2 = Conv2D(16, (3,3),activation = 'relu',kernel_initializer='he_uniform',
#                                     kernel_regularizer=regularizers.l2(self.l2Lambda),
#                                     data_format='channels_last',name='convlayer2')(batchNorm1)
#                                 
#         
#         batchNorm2 = BatchNormalization(name='batchNorm2')(conv2)
#         #3rd convolution Layer
#         conv3 = Conv2D(10, (3,3), activation = 'relu',kernel_initializer='he_uniform',
#                                     kernel_regularizer=regularizers.l2(self.l2Lambda),
#                                     data_format='channels_last',name='convlayer3')(batchNorm2)
#                                 
#             
#         batchNorm3 = BatchNormalization(name='batchNorm3')(conv3)
#         maxpool3 = MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding='same',name = 'pool3')(batchNorm3)
#         drop3 = Dropout(0.25,name='dropout3')(maxpool3)
#         
#         #adding LSTM layer
#         #reshape = Reshape(target_shape=)
#         flatForLSTM = TimeDistributed(Flatten(name='flatLayer_for_LSTM'))(drop3)
#         lstm0 = LSTM(25,activation='tanh',return_sequences=True,dropout=0.25,
#                      recurrent_dropout=0.25,recurrent_initializer='glorot_uniform',kernel_regularizer = regularizers.l2(self.l2Lambda),
#                      recurrent_regularizer = regularizers.l2(self.l2Lambda),name='lstm_0')(flatForLSTM)
# =============================================================================
        #lstm1 = LSTM(10,activation='tanh',return_sequences=True,dropout=0.25, name='lstm_1')(lstm0)
# =============================================================================
        
        #Flatten layer
        flat1 = Flatten(name='flat1')(lstm0)
        
# =============================================================================
#     sending the same image to Deep Neural netword parellely
# =============================================================================
            
        flatIp = Flatten(name='FlattenInput')(input1)
        denseIP1 = Dense(1024, activation = 'sigmoid',kernel_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l2(self.l2Lambda), name='denseInput1')(flatIp)
        dropDense1 = Dropout(0.25,name='DenseDrop1')(denseIP1)
        
        denseIP2 = Dense(1024, activation='sigmoid',kernel_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l2(self.l2Lambda),name='input2_dense1')(dropDense1)
        dropDense2 = Dropout(0.25,name='DenseDrop2')(denseIP2)
    # =============================================================================
    #   Concatinate both Branches
    # =============================================================================
        
        concat = concatenate([flat1,dropDense2])
        
    # =============================================================================
    # Fully connected layers
    # =============================================================================
        hidden1 = Dense(1024,activation='relu',kernel_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l2(self.l2Lambda), name='FUll_1')(concat) 
        drop6 = Dropout(0.25,name='dropout6')(hidden1)
        
        hidden2 = Dense(1024,activation='relu',kernel_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l2(self.l2Lambda), name='FUll_2')(drop6) 
        drop7 = Dropout(0.25,name='dropout7')(hidden2)
        
        output = Dense(self.numCategories, activation='softmax', name='FUll_3')(drop7) 
        #drop5 = Dropout(0.2,name='dropout4')(hidden2)
        
        self.model = Model(inputs=input1,outputs=output)
        self.baseModel = self.model
        #self.model.compile(optimizer=optim,loss='categorical_crossentropy',metrics=['accuracy'])
        #self.model = multi_gpu_model(self.model,gpus=1)
        self.model.compile(optimizer=optim,loss='categorical_crossentropy',metrics=['accuracy'])
        print(self.model.summary())
        
        plot_model(self.model, to_file= self.path +'Models/Seperate_Models/' + 'modelStructure_CNN+LSTM.png')
        
        print('Model configured..!')
        #return self.model, self.train_generator.samples, self.validation_generator.samples
        
    # =============================================================================
    #   Training The model
    # =============================================================================
    def training(self):
        stepsPerEpochs = self.train_generator.samples//self.batchSize
        #stepsPerEpochs = self.validation_generator.samples//self.batchSize
        validationSteps = self.validation_generator.samples//self.batchSize        
        #with tf.device('/gpu:1'):
        
# =============================================================================
#         config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
#         sess = tf.Session(config=config) 
#         keras.backend.set_session(sess)
# =============================================================================
        self.hist = self.model.fit_generator(self.train_generator,
                            steps_per_epoch =stepsPerEpochs,
                            epochs=self.epochs,
                            validation_data = self.validation_generator,
                            validation_steps = validationSteps,
                            verbose = 1,
                            callbacks = [self.reduceLR, 
                                        self.EarlyCheckPt,
                                        self.ModelCkPt]
                                                #AltModelCheckpoint(self.path+'Models/OCR_Epochs.h5',self.baseModel)] 
                                    )
            #except Exception as e:
                #print("Got issues : ", e)
               
        #saving the model after final Epoch
        save_model(self.model,self.path+'Models/Seperate_Models/Final_Alphabets_v3.h5')
        self.model.set_weights(self.model.get_weights())
        self.model.save(filepath=self.path+'Models/Seperate_Models/Final_Alphabets_v3.h5')        
        N = np.arange(0, self.epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, self.hist.history["loss"], label="train_loss")
        plt.plot(N, self.hist.history["val_loss"], label="val_loss")
        plt.plot(N, self.hist.history["acc"], label="train_acc")
        plt.plot(N, self.hist.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy (Simple NN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(self.path + 'OutputAlphabets_v3.png')



