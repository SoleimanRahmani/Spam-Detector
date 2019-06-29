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

### Exploring the Dataset

```python
df = pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
```

### Features and Labels

```python
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
```

And part of dataset is as follows:

```python
 class                                            message  label
0   ham  Go until jurong point, crazy.. Available only ...      0
1   ham                      Ok lar... Joking wif u oni...      0
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...      1
3   ham  U dun say so early hor... U c already then say...      0
4   ham  Nah I don't think he goes to usf, he lives aro...      0
5  spam  FreeMsg Hey there darling it's been 3 week's n...      1
6   ham  Even my brother is not like to speak with me. ...      0
7   ham  As per your request 'Melle Melle (Oru Minnamin...      0
8  spam  WINNER!! As a valued network customer you have...      1
9  spam  Had your mobile 11 months or more? U R entitle...      1
```

### # Extract Feature With CountVectorizer

```python

cv = CountVectorizer()
X = cv.fit_transform(X)  # Fit the Data

```

### split our data set in training set and test set

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print([np.shape(X_train),np.shape(y_train), np.shape(X_test),np.shape(y_test)])
```
```python
Output:
[(3900, 8672), (3900,), (1672, 8672), (1672,)]
```

### Machine Learning Model Selection 
In this step, several classifiers are implemented and their performance is compared to each other. The machine learning methods that used in ```check.py``` file are:

* gradient boosting Classifier
* SVC
* Decision Tree Classifier
* Random Forest Classifier
* Naive Bayes classifier

```python
# Classifiers
classifiers = []
model1 = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
            max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
            n_estimators=100, n_jobs=1, nthread=None,
            objective='binary:logistic', random_state=0, reg_alpha=0,
            reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
            subsample=1, verbosity=1)
classifiers.append(model1)

model2 = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
            kernel='rbf', max_iter=-1, probability=False, random_state=None,
            shrinking=True, tol=0.001, verbose=False)
classifiers.append(model2)

model3 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
classifiers.append(model3)

model4 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=True, random_state=42, verbose=0,
            warm_start=False)
classifiers.append(model4)

model5 = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
classifiers.append(model5)
```
To compare their performance, their accuracy score and confusion matrix is obtained:

```python
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy of %s is %s" % (clf, acc))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of %s is %s" % (clf, cm))
```

And the result is as follows:

```python
Accuracy of XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1) is 0.965311004784689
--------------------------------------------------------------------------
Confusion Matrix of XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1) is 
       [[1446    7]
       [  51  168]]



Accuracy of SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False) is 0.8690191387559809
--------------------------------------------------------------------------
Confusion Matrix of SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False) is 
       [[1453    0]
       [ 219    0]]



Accuracy of DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best') is 0.9659090909090909
--------------------------------------------------------------------------
Confusion Matrix of DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best') is 
            [[1430   23]
            [  34  185]]



Accuracy of RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=True, random_state=42, verbose=0, warm_start=False) is 0.9730861244019139
--------------------------------------------------------------------------
Confusion Matrix of RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=True, random_state=42, verbose=0, warm_start=False) is 
            [[1452    1]
            [  44  175]]



Accuracy of MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) is 0.9784688995215312
--------------------------------------------------------------------------
Confusion Matrix of MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) is 
[[1435   18]
 [  18  201]]

```


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




