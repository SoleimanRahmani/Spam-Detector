# Spam-Detector
A simple Flask API to detect spam or ham messages using following Python packages:

* Sklearn
* Pandas
* Text Extraction

## Description
The purpose of this repository is to build a web app which classifies spam or ham (non-spam) messages. The goal of this project is to first, build the best machine learning model offline. In the next step, make the model available as a service by creating an API for the model using Flask. Finally, use the service to predict online where the user can submit a message for classification.


## Building Machine Learning Model
In the first step, several message classifiers have been implemented and compared in ```check.py``` file. The original meta-data file is a collection of messages tagged as spam or ham that can be found in Keggle. The goal in this stage is to use this dataset to build a prediction model that will accurately classify which texts are spam or ham.

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import tree
from mleap.sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgboost
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
```

The machine learning methods that used in ```check.py``` file are:

* gradient boosting Classifier
* SVC
* Decision Tree Classifier
* Random Forest Classifier
* Naive Bayes classifier



Naive Bayes classifiers are a popular statistical technique of e-mail filtering. They typically use bag of words features to identify spam e-mail. 






In the second step, 







```bash
pip install foobar
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```





<img width="824" alt="Screen Shot 2019-06-26 at 8 35 22 PM" src="https://user-images.githubusercontent.com/45254300/60224871-5a39fa00-9852-11e9-96ac-33d915bdd4a0.png">




