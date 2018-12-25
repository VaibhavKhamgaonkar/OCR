# OCR
For hand written text identification (OCR)

To refer the Data set please use the Following Dataset :
**https://www.kaggle.com/vaibhao/handwritten-characters**

THe dataset contains 39 categories including :
1. Alphabtes (small and caps merged together just to avoid mis classification) -- > 26 
2. Digits (1 to 9) : Digit 0 is added to character O for avoiding misclassification
3. Some special characters which include &, #, $, @

**The data set contains Train and Validation folders containing 0.8+ Milion Training Records and 20,000 + validation records**
**ALl image are of 32,32 pixel black and white image..**

The arechitecture used to train the model is CNN along with deep learning network running paraller to CNN:
 Please refer the Flow diagram below :

![Model_Architecture](https://github.com/VaibhavKhamgaonkar/OCR/blob/master/modelStructure_CNN%2BLSTM.png)
 

* Added Requirment files. Use 

      pip install -r /path/to/requirements.txt #for installation of requirements.


**Description: -**
1. For Inference : 

* Put all your image documents to the **SampleTestForms** which you want to extract infomation from image. 
* getText.py and ParseDocument_v2.py these 2 files sould be there in same folder as these files are used to identifying the Text from the scanned image document.
* configure the ParseDocument_v2.py file wiht the Model Path and label paths. The default location of all models and label files are inside Model folder. please change if you are storing these files elsewhere.

* Run getText.py file in command prompt ==> Program will fetch the hand writtern characters, print it on screen and simultaneoulsy it will create a csv file with all the informations in Output directory(located inside current working directory).
* all the cropped and processed images will be there inside the RuntimeImages folder for troubleshooting..(if required)


2. For Model Training:

* The Data should be present in in folder structure.

i.e.

     * ParentFolder 
     * |--Train ----
     * |        ----
     * |        ----
     * |
     * |--Validation ----
     * |             ----
     * |             ----


* Edit the CNN_mainScript.py file and configure the Training data path, Validation Data path, Number of classes, Learning rate, batchSize, regularization parameter.

* Edit the LSTM_Training.py file and update the Following parameters

1. Model checkpoint == Enter the name of model and path to which the Model get saved after each epochs if validation loss is reduced.
2. Label File Name
3. Update the details at the bottom of the file.




**Note**: _This OCR model is tuned for the sample forms added in the SampleTestForm folder. If you are trying for something different than the mentioned form then you have to tune the code in order to incorporate new changes ..._

Check for the demostration :
https://www.linkedin.com/feed/update/urn:li:activity:6454087778501259264 
