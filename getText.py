#Main File

import cv2,pandas as pd
import numpy as np, os, time,pickle
from ParseDocument_v2 import Document
from keras.models import load_model

counts = 0
global X,Y,W,H



allOutput = []
#dfNames = ['FirstName','LastName', 'Email', 'Street','City','State','ZipCode','Phone','BirthDay']
path = os.path.dirname(os.path.abspath('__file__')) + '/'
#for batch mode
batchFilepath = path + 'SampleTestForms/'

images = [img for img in os.listdir(batchFilepath) if '.png' in img or '.jpg' in img]#'.DS_Store' not in img]
#images =[img for img in images if '.png' in img or '.jpg' in img]
print('==============================================')
print('Found {} images...'.format(len(images)))
print('==============================================\n')
#get the input image
for imageFile in images:
    print('==============================================')
    print('Starting ', imageFile)
    print('==============================================')
    imgPath = batchFilepath + imageFile
    image = cv2.imread(imgPath)
    wd,ht = image.shape[:2]
    #print(wd)
    #print(ht)
    #image = cv2.resize(image,(1500,1345),cv2.INTER_AREA)
    '''if ht not in range(1300,1500): 
        image = cv2.resize(image,(1500,1100),cv2.INTER_AREA)
        print(image.shape)'''
    img = image.copy()

    obj = Document()
    image, imgCnt = obj.processedImage(image)

    image_with_lines = obj.dilateImage(imgCnt.copy(),150)

    contours = obj.getCountours(image_with_lines.copy())
        #cv2.imwrite(baseDir +'/testDir/' + 'afterCnt1.jpg', image_With_Lines) 
    contours = obj.sortCountours(contours, "top-to-bottom")

    #for storing the output initialising a emppty list and counter
    output = []
    i = 0
    predictedText = []
    #getting the inmages line by line from each contours
    for iter, line_Area in enumerate(contours):
        #print('Iteration:- ', iter)
        #print('len of Contours:- ', len(contours))
        counts = counts + 1
        x,y,w,h = cv2.boundingRect(line_Area)

        X,Y,W,H = x,y,w,h
        line_Image = imgCnt[y:y+h, x:x+w]
        #path = r'C:\Users\sachin\Desktop\Images\\' + str(counts)+'.png'
        
        line_Contours = obj.getCountours(imgCnt[y:y+h, x:x+w])
        line_Contours = obj.sortCountours(line_Contours,"left-to-right")
        #getting the Text 
        text = obj.getTextFromImage(image[y:y+h, x:x+w], line_Contours, Width=8, Height=5)   
        print(text)
        predictedText.append(text)
    #print('Len of predicted text = ',len(predictedText))
    while i < len(predictedText):
        x = ' '.join(predictedText[i].split(' ')[1:])
        #x = ' '.join(x)
        if len(contours)>9 and i == 2:
            #handling for .com text which comes at 4rth place
            x = x + ' '.join(predictedText[i+1])
            i+=1
        #appending the Extratced Text into 1st output varibale
        output.append(x)
        i+=1
    #print('len of output',len(output))
    print('==============================================')
    print('Done with Processing ', imageFile)
    print('==============================================')
    #Now updating it to final List
    allOutput.append(output) 
print(len(allOutput))
ret = obj.storeData(allOutput)
if ret == True:
    print('Done with Scanning.. Result can be found at {}'.format(path + 'Output/Output.csv') )
else:
    print('Something went wrong while saving the csv file..!!!')





    #break



