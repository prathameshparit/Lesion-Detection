# Periapical Lesions Detection



Dental caries is one of the most prevalent and disastrous diseases in dentistry. The
traditional methods involving observations of x-ray images and human judgments lead to failure
due to a lack of adequate and accurate data. The recent advances in A.I. have been a great help in
generic dental disease diagnosis. However, few researchers have paid due attention to the
applications of A.I. in the diagnosis of dental caries and root canal procedures. It is proposed to
develop an A.I. model that will enhance image interpretation precision, leading to proper
techniques and lesser failures

## Demo

https://user-images.githubusercontent.com/63944541/209447881-b09ab52d-ba58-490b-9b88-50082b57b883.mov

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


