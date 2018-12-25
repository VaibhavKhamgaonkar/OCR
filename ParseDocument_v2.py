#main Scriptfor loading the Text Images and fetching the text out of the scanned document

import keras
import cv2, numpy as np, pandas as pd, os, time 
import pickle
from keras.models import load_model
import imutils

#path = os.path.dirname(os.path.abspath('__file__')) + '/'


class Document():
    def __init__(self):
        self.dfNames = ['FirstName','LastName', 'Email', 'Street','City','State','ZipCode','Phone','BirthDay']
        #self.modelPath = modelPath
        self.path = os.path.dirname(os.path.abspath('__file__')) + '/' 
        self.path = self.path.replace('\\','/')
        alphaNumnericModel_path = self.path + 'Models/AlphabetNumeric_v3.h5'
        alphabetModel_path = self.path + 'Models/Alphabets_v3.h5'
        digitModel_path = self.path + 'Models/Digit_v2.h5'
        atr_Model_path = self.path + 'Models/@.h5'
        _modelpath = self.path + 'Models/_model.h5'
        
        """#This is General model for everything(alpabets,digits and Special characters"""
        self.alphaNumericModel = load_model(alphaNumnericModel_path)
        self.alphabetModel = load_model(alphabetModel_path)
        self.digitModel = load_model(digitModel_path)
        self.atrModel = load_model(atr_Model_path)
        self._model = load_model(_modelpath)

        self.model = None
        self.label = None
        self.checkForATR = False
        #getting the labels from pickled file #CNN_OCRModel_v1.h5
        with open (self.path + 'Models/AlphabetNumericLabels_v3.pkl','rb') as f:
            self.alphaNumericLabels = pickle.load(f)
        self.alphaNumericLabels = dict([(v,k) for k,v in self.alphaNumericLabels.items()])
        
        with open (self.path + 'Models/AlphabetsLabels_v3.pkl','rb') as f:
            self.alphabetLabel = pickle.load(f)
        self.alphabetLabel = dict([(v,k) for k,v in self.alphabetLabel.items()])

        with open (self.path + 'Models/Digit_v2.pkl','rb') as f:
            self.digitLabel = pickle.load(f)
        self.digitLabel = dict([(v,k) for k,v in self.digitLabel.items()])

        with open (self.path + 'Models/@label.pkl','rb') as f:
            self.atrLabel = pickle.load(f)
        self.atrLabel = dict([(v,k) for k,v in self.atrLabel.items()])
        
        #getting the labels from pickled file #CNN_OCRModel_v1.h5
        with open (self.path + 'Models/_Labels.pkl','rb') as f:
            self._label = pickle.load(f)
        self._label = dict([(v,k) for k,v in self._label.items()])


        self.list_Character_Positions = []
        self.count = 0

    # Function for Document processing
    def processedImage(self,image):
        img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img,(7,7),0)
        ret, thres = cv2.threshold(img.copy(),0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        #ret, thres = cv2.threshold(thres.copy(),0,255,cv2.THRESH_BINARY_INV )#| cv2.THRESH_BINARY_INV)
        cv2.imwrite(self.path + 'EvalImages/' + 'ProcessedImage.png', thres)
        thres = cv2.erode(thres,np.ones((3,3),np.uint8),iterations=1)
        cv2.imwrite(self.path + 'EvalImages/' + 'ProcessedImage1.png', thres)
        thres_process = cv2.dilate(thres,np.ones((3,3),np.uint8),iterations=1)
        cv2.imwrite(self.path + 'EvalImages/' + 'ProcessedImage2.png', thres_process)
        #close = cv2.morphologyEx(thres.copy(),cv2.MORPH_OPEN, np.zeros((5,5),dtype = 'uint8'))

        #thres_forContours = cv2.adaptiveThreshold(thres_process.copy(),255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,3,10 )
        #cv2.imwrite(self.path + 'EvalImages/' + 'ProcessedImage3.png', thres_forContours)
        #erd = cv2.erode(close,np.zeros((3,3),dtype = 'uint8'))
        #cv2.imwrite(self.path + 'EvalImages/' + 'ProcessedImage3.png', thres_forContours)
        # compute gradients along the X and Y axis, respectively
        '''gX = cv2.Sobel(thres_process.copy(), ddepth=cv2.CV_64F, dx=0, dy=1)
        gY = cv2.Sobel(thres_process.copy(), ddepth=cv2.CV_64F, dx=1, dy=0)
        
        # the `gX` and `gY` images are now of the floating point data type,
        # so we need to take care to convert them back to an unsigned 8-bit
        # integer representation so other OpenCV functions can utilize them
        gX = cv2.convertScaleAbs(gX)
        gY = cv2.convertScaleAbs(gY)
        
        # combine the sobel X and Y representations into a single image
        sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)'''
        #dilating the image to capture entire letter
        #thres_process_dilate = cv2.dilate(thres_forContours.copy(), np.ones((3,3),np.uint8),iterations=2)
        #cv2.imwrite(self.path + 'EvalImages/' + 'ProcessedImage4.png', thres_process_dilate)
        autoCanny = imutils.auto_canny(thres_process.copy(),sigma = 0.33)
        cv2.imwrite(self.path + 'EvalImages/' + 'ProcessedImage3.png', autoCanny)

        return thres_process, autoCanny

    def dilateImage(self, image, number):
        kernel = np.ones((9,number),np.uint8) #number : represents wideness of the dilation. i.e for line level or word level or character level.
        img_dilation = cv2.dilate(image, kernel, iterations=3)
        return img_dilation

    def getCountours(self,input_Image):
        #cv2.imwrite(baseDir +'/testDir/' + '1stContour.jpg', input_Image) 
        temp_image, contours, hierarchy = cv2.findContours(input_Image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.imwrite(baseDir +'/testDir/' + '2ndContour.jpg', input_Image) 
        #cv2.imwrite(baseDir +'/testDir/' + 'tempImage.jpg', temp_image) 
        return contours


    def sortCountours(self,cnts, method="left-to-right"):
        # initializing the reverse flag and sorting index
        reverse = False
        i = 0
        # handling the flag if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b:b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (cnts)

    def Draw_Contours(self, image, contours):
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            #print (str(x) + " - " + str(y)+ " - " + str(w) + " - " + str(h))
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)


    def getNewResizedImage(self, input_Image, image_Size):
        height,width = input_Image.shape
        #print (height, width)

        if width > height:
            aspect_Ratio = (float)(width/height)
            width = 22
            height = round(width/aspect_Ratio)
        else:
            aspect_Ratio = (float)(height/width)
            height = 22
            width = round(height/aspect_Ratio)
            
        input_Image = cv2.resize(input_Image, (width,height), interpolation = cv2.INTER_AREA )
        
        height,width = input_Image.shape
        
        number_Of_Column_To_Add = 30-width
        temp_Column = np.zeros( (height , int(number_Of_Column_To_Add/2)), dtype = np.uint8)
        input_Image = np.append(temp_Column, input_Image, axis=1)
        input_Image = np.append(input_Image, temp_Column, axis=1)


        height,width = input_Image.shape
        number_Of_Row_To_Add = 30-height
        temp_Row= np.zeros( (int(number_Of_Row_To_Add/2) , width ), dtype = np.uint8)
        input_Image = np.concatenate((temp_Row,input_Image))
        input_Image = np.concatenate((input_Image,temp_Row))

        return cv2.resize(input_Image, (image_Size,image_Size), interpolation = cv2.INTER_AREA )


    def getTextFromImage(self, image, contours, Width=5, Height=5):
        global count,X,Y,W,H
        alphabetPrediction = ''
        count = 0
        #print('Entered')
        #image = np.array(image,dtype='uint8')
        
        Word_Dilated_Image = Document.dilateImage(self,image,18)
        '''cv2.imshow('image',Word_Dilated_Image)
        cv2.waitKey(0)'''
        #cv2.imwrite(self.path + 'EvalImages/Dilate.png', Word_Dilated_Image)

        Word_Contours = Document.getCountours(self,Word_Dilated_Image)
        Word_Contours = Document.sortCountours(self,Word_Contours,"left-to-right")
        last_Word_Contour_Index = 0
        Word_X, Word_Y, Word_W, Word_H = cv2.boundingRect(Word_Contours[last_Word_Contour_Index])
        last_Word_Contour_Max_X_Range = Word_X + Word_W
        condition = 0
        spaceFlag = False
        
        for k,cnt in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cnt)
            arc = cv2.arcLength(cnt,True)
            #print('contour {} with area is : {}'.format(cnt, area))

            #if k == 0:
            give_Space = False
            #condition = 0
            #if w > Width and h > Height :# and x > 0 and y > 0:
            if w>=Width and h > Height and arc > 80:
                if condition == 0:
                    condition += 1
                #checking the model for prediction
                    firstLetter = cv2.resize(image[y-1:y+h+1, x-1:x+w+1],(35,35),cv2.INTER_AREA)
                    firstLetter = Document.getNewResizedImage(self,firstLetter , 32)
                    predFirstLetterProb = self.alphabetModel.predict(firstLetter.reshape(1,32,32,1))[0]
                    index = int(np.argmax(predFirstLetterProb))
                    firstLetter = self.alphabetLabel[index]
                    #print('First Letter is : ', firstLetter)

                    #now checking the correct model condition
                    if str(firstLetter).strip() in ['Z','P', 'B']:
                        #print('Got Digits..')
                        self.model = self.digitModel
                        self.label = self.digitLabel
                        flag = 'digit'
                        check = True
                        self.checkForATR = False
                        #print('Labels are : ', self.digitLabel)

                    elif firstLetter in ['U','C','F','L','M',  'N', 'W']:
                        self.model = self.alphabetModel
                        self.label = self.alphabetLabel
                        flag = 'alphabet'
                        check = True
                        self.checkForATR = False

                    else:
                        self.model = self.alphaNumericModel
                        self.label = self.alphaNumericLabels
                        self.checkForATR = True
                        flag = 'alphanumeric'
                        check = True

                ## Spacing based on word formation
                if x > last_Word_Contour_Max_X_Range:
                    give_Space = True
                    spaceFlag = True
                    last_Word_Contour_Index = last_Word_Contour_Index + 1
                    Word_X, Word_Y, Word_W, Word_H = cv2.boundingRect(Word_Contours[last_Word_Contour_Index])
                    last_Word_Contour_Max_X_Range = Word_X + Word_W
                    
                if give_Space == True:
                    alphabetPrediction = alphabetPrediction + " "
                    self.list_Character_Positions.append((-1,-1,-1,-1," "))
                #resize_image = Document.getNewResizedImage(self,image[y:y+h, x:x+w] , 35)
                #resize_image[resize_image>0] = 240        
                #resize_image = cv2.resize(resize_image,(32,32),cv2.INTER_AREA)
                resize_image = cv2.resize(image[y-1:y+h+1, x-1:x+w+1],(35,35),cv2.INTER_AREA)

                resize_image = Document.getNewResizedImage(self,resize_image, 32)
                #resize_image[resize_image>80] = 240
                #resize_image = cv2.erode(resize_image,np.zeros((3,3),dtype='uint8'),iterations=1)
                #cpath = baseDir + '/testDir/' + str(count)+'.png'
               
                self.count+=1
                #adding check to selct character model for headings
                if check == True:
                    #print('Flag is : ', flag)
                    if give_Space == False and spaceFlag == False:
                        self.model = self.alphabetModel
                        self.label = self.alphabetLabel
                    else:
                        #if give_Space == True:
                        if flag == 'alphabet':
                            self.model = self.alphabetModel
                            self.label = self.alphabetLabel
                            check = False
                            self.checkForATR = False

                        elif flag == 'digit':
                            #print('DIGIT Model Selected..!!')
                            self.model = self.digitModel
                            self.label = self.digitLabel
                            check = False
                            self.checkForATR = False
                        else:
                            #print('AlphaNumeric Model Selected..!!')
                            self.model = self.alphaNumericModel
                            self.label = self.alphaNumericLabels
                            check = False
                    '''else:
                        self.model = self.alphabetModel
                        self.label = self.alphabetLabel'''
                        #check = False
                #checking for "_"
                #_img = Document.getNewResizedImage(self,resize_image, 32)
                #
                #_img = cv2.resize(_img,(32,32),cv2.INTER_AREA)
                #_img = cv2.erode(_img,np.ones((3,3),dtype='uint8'),iterations=1)
                cv2.imwrite(self.path + 'EvalImages/' + str(self.count)+'.jpg', resize_image)

                charProb = self._model.predict(resize_image.reshape(1,32,32,1))[0]
                #charProb = self._model.predict(resize_image.reshape(1,32,32,1))[0]
                index = int(np.argmax(charProb))
                char = self._label[index]

                if str(char).lower() == 'junk':
                    if self.checkForATR == True:
                        charProb = self.atrModel.predict(resize_image.reshape(1,32,32,1))[0]
                        index = int(np.argmax(charProb))
                        char = self.atrLabel[index]
                        #print(char)
                        '''cv2.imshow('resImag',resize_image)
                        cv2.waitKey(0)'''
                        if str(char).lower() == 'junk':
                            
                            #print('Proceeding for general categories of caracters.')
                            prob = self.model.predict(resize_image.reshape(1,32,32,1))[0]
                            index = int(np.argmax(prob))
                            character = self.label[index]
                            
                            alphabetPrediction = alphabetPrediction + character
                            self.list_Character_Positions.append(alphabetPrediction)
                        else:
                            alphabetPrediction = alphabetPrediction + char
                            self.list_Character_Positions.append(alphabetPrediction)
                    else:
                        prob = self.model.predict(resize_image.reshape(1,32,32,1))[0]
                        index = int(np.argmax(prob))
                        character = self.label[index]
                        '''if flag == 'digit':
                                print('length of Labels is :',len(self.label))'''
                        alphabetPrediction = alphabetPrediction + character
                        self.list_Character_Positions.append(alphabetPrediction)

                else:
                    #print('Found Underscore : ', char)
                    alphabetPrediction = alphabetPrediction + char
                    self.list_Character_Positions.append(alphabetPrediction)
            #Checking _ character
            elif w in range(2,Width) and h <= Height:
                check = True
                
                resize_image = Document.getNewResizedImage(self,image[y-1:y+h+1, x-1:x+w+1], 32)
                #resize_image = cv2.resize(resize_image,(32,32),cv2.INTER_AREA)
                
                #resize_image = cv2.erode(resize_image,np.zeros((3,3),dtype='uint8'),iterations=2)
                #cv2.imwrite(self.path + 'EvalImages/' + str(self.count)+'.png', resize_image)
                #cv2.imshow('resize_image',resize_image)
                #cv2.waitKey(0)
                charProb = self._model.predict(resize_image.reshape(1,32,32,1))[0]
                index = int(np.argmax(charProb))
                char = self._label[index]
                if str(char).lower() != 'junk':
                    alphabetPrediction = alphabetPrediction + char
                    self.list_Character_Positions.append(alphabetPrediction)
            else:
                pass
                #check = True
                #flag = None

                '''count = count + 1          
                if checkFor == 'Letters':
                    prob = model.predict_proba(resize_image.reshape(1,32,32,1))[0]
                    sort_alphabet_probability = -np.sort(-prob)
                    #if sort_alphabet_probability[0] >= 0.05:
                    temp_Index = int(model.predict_classes(resize_image.reshape(1,28,28,1)/255.0)[0])
                    alphabetPrediction = alphabetPrediction + list({k for k,v in y_labels.items() if v == temp_Index})[0]
                    #list_Character_Positions.append((x+X,y+Y,w,h,str(list({k for k,v in y_labels.items() if v == temp_Index})[0])))
                        
                else:
                    temp_Index = int(model.predict_classes(resize_image.reshape(1,28,28,1)/255.0)[0])
                    alphabet_probability = (model.predict_proba(resize_image.reshape(1,28,28,1)/255.0))
                    sort_alphabet_probability = -np.sort(-alphabet_probability)
                    #if sort_alphabet_probability[0,0] > 0.95:
                    alphabetPrediction = alphabetPrediction + digitList[int(temp_Index)]'''
                    #list_Character_Positions.append((x+X,y+Y,w,h,str(digitList[int(temp_Index)])))

        return alphabetPrediction


    def storeData(self, data):
        
        try:
            df = pd.DataFrame(data)
            df.columns = self.dfNames
            df.to_csv(self.path + 'Output/Output.csv',index=False)
            return True
        except Exception as e:
            print('Exception... ', e)
            return False


            





