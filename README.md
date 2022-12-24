# Periapical Lesions Detection



Dental caries is one of the most prevalent and disastrous diseases in dentistry. The
traditional methods involving observations of x-ray images and human judgments lead to failure
due to a lack of adequate and accurate data. The recent advances in A.I. have been a great help in
generic dental disease diagnosis. However, few researchers have paid due attention to the
applications of A.I. in the diagnosis of dental caries and root canal procedures. It is proposed to
develop an A.I. model that will enhance image interpretation precision, leading to proper
techniques and lesser failures

### The project includes these points for Periapical Lesions Detection:
- Comparative analysis of lesion detection methods for automated Periapical Lesions Detection.
- Prepared a dataset for Periapical Lesions Detection using python with ML.
- A novel model for Periapical Lesions Detection diagnosis.

##### About:

Peripheral giant cell granuloma is a common tumor-like lesion of the oral cavity that conveys a reactive response in the periodontium, periodontal ligament and gingivae. Our website predicts following things:

- Detect periapical lesions and root fractures to improve the diagnosis.
- Determines the working length measurements in the endodontic treatment for better results.
- Predicts whether endodontic retreatment will succeed or an alternative has to be explored.

It occurs exclusively on the edentulous alveolar ridge and gingivae as a smooth, reddish-blue, pedunculated, sessile, or nodular mass, which is firm to palpation. In some cases the clinical appearance of the lesion is similar to pyogenic granuloma; however peripheral giant cell granuloma is more bluish-purple colored as compared with bright red color of pyogenic granuloma

##### The Project Include 4 things:

1. Periapical Lesions Detection - Determines the difficulty level of the case.
2. Root Fractures Detection - Detects periapical lesions and root fractures to improve the diagnosis.
3. Working length measurements - Determines the working length measurements in the endodontic treatment for better results.
4. Endodontic retreatment outcome - Predicts whether endodontic retreatment will succeed or an alternative has to be explored.


## Exploratory Data Analysis

- 100: Total number of images model is trained on Lesion

- 100: Total number of images model is trained on Normal


## Data Visualization
![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/analysis.png?raw=true)

## Features

- Drag and drop images 
- Drop images for Periapical Lesions Detection
- It predicts the input image for 2 classes(Lesion, Normal)
- Predicts the accuracy with 6 different classifiers for the input image
- Each Classifier provides the metrics for particular prediction with graph



## Tech

The website uses a number of open source projects to work properly:

- [Tensorflow] - Deep learning application framework
- [Scikit-Learn] - Bank for classification, predictive analytics, and very many other machine learning tasks.
- [Flask] - Framework for creating web applications in Python easier.
- [Matplotlib] - A low level graph plotting library in python that serves as a visualization utility.
- [Numpy] - Used for working with arrays
- [Pandas] - Used for data analysis and associated manipulation of tabular data in Dataframes

## Screenshots and Steps

**1. Landing Page:**

- This is the landing page for the web application 

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/Landing.png?raw=true)

**2. Upload button:**
 
- Later on the web application it provides 3 different buttons along with a upload button where you upload your input image and later it provides you with 3 buttons of Preprocessing, Feature Extraction and Prediction of the uploaded image

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/upload.png?raw=true)

**3. Preprocessing:**


- After uploading the image the image needs to be preprocessed where it is preprocessing using two techniques which is Resizing of the uploaded input image from it's original size to the size which is required for the image to predict on.

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/resize.png?raw=true)

- After resizing of the image Data Augmentation is applied on the input image  
- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/augment.png?raw=true)

**5. Feature Extraction :**

- After Preprocessing comes the part of Feature Extraction where we extract important features of the input image by converting the uploaded image from Original to Grayscale and pointing out the important parts required for the model to predict the following image

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/resize.png?raw=true)

**6. Classifiers :**

- After the Feature Extraction comes the part of Prediction where the project is trained on 7 different classifiers which are SVM, KNN, ANN, DT, CNN, Hybrid(SVM+ANN) and it displays it's prediction on those 7 classifiers along with the confidence at which it has predicted the following image 
- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/results.png?raw=true)

- If you click on any of the classifiers it further shows you the classification metrics on that particular classifier along with the visualization of that model
-![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/metrics.png?raw=true)

**7. Comparitive Analysis :**
- At last the application provides you with a comparitive analysis of all the classifiers where you can compare the accuracy of each classfier side by side in the format of table as well as graph
- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/comparitive.png?raw=true)

## Results

The following project has shown some promising results for classifying the Periapical Lesions Detection into 2 classes which is Normal, Lesion and here's the classification for the following

![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/151be162b346f33426059f2304075ee56923027d/readme%20images/potato/graph.png?raw=true)




## Installation

Website requires these steps to install the application on your device


On terminal:

Download virtual env library
```sh
pip3 install -U pip virtualenv
```

Create a virtual environment on your device
```sh
virtualenv  -p python3 ./venv
```

Download all the dependencies provided on requirements.txt
```sh
pip install -r .\requirements.txt
```

Activated the virtual environment
```sh
.\pp\Scripts\activate
```

Run app.py after completing all the steps.





[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   
[Tensorflow]: <https://www.tensorflow.org/>
[Scikit-Learn]: <https://scikit-learn.org/stable/>
[Flask]: <https://flask.palletsprojects.com/en/2.1.x/>
[Matplotlib]: <https://matplotlib.org/>
[Numpy]: <https://numpy.org/>
[Pandas]: <https://pandas.pydata.org/>


