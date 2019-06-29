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

Confusion Matrix of RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=True, random_state=42, verbose=0, warm_start=False) is 
            [[1452    1]
            [  44  175]]



Accuracy of MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) is 0.9784688995215312

Confusion Matrix of MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) is 
[[1435   18]
 [  18  201]]

```
As it can be seen, best accuracy is achieved with the popular statistical technique, Naive Bayes classifier.
In order to obtain the best Naive Bayes model, different models are trained hanging the regularization parameter (α) and the accuracy, recall and precision of the model with the test set are evaluated.

```python

list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1
matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns =
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
print(models.head(n=10))
```

```python
Output:

 alpha  Train Accuracy       ...        Test Recall  Test Precision
0  0.00001        0.998462       ...           0.904110        0.925234
1  0.11001        0.996667       ...           0.931507        0.918919
2  0.22001        0.996154       ...           0.931507        0.910714
3  0.33001        0.995128       ...           0.931507        0.906667
4  0.44001        0.995128       ...           0.931507        0.902655
5  0.55001        0.995128       ...           0.926941        0.902222
6  0.66001        0.995128       ...           0.926941        0.906250
7  0.77001        0.994872       ...           0.926941        0.906250
8  0.88001        0.994872       ...           0.926941        0.910314
9  0.99001        0.994359       ...           0.917808        0.917808

```

And the model with best can be obtained:

```python
print('The model with best test Accuracy')
best_index = models['Test Accuracy'].idxmax()
print(models.iloc[best_index, :])
```
```python
Output:
The model with best test accuracy
alpha             0.110010
Train Accuracy    0.996667
Test Accuracy     0.980263
Test Recall       0.931507
Test Precision    0.918919
Name: 1, dtype: float64
````
Below, the model with test precision higher than 90% and highest test accuracies is shown:

```python
best_index = models[models['Test Precision']>=0.9]['Test Accuracy'].idxmax()
```
```python
Output:
 alpha             0.110010
Train Accuracy    0.996667
Test Accuracy     0.980263
Test Recall       0.931507
Test Precision    0.918919
Name: 1, dtype: float64

```
Hence, the best model is obtained and its confusion matrix is shown below:

```python
best_index = models[models['Test Precision']>=0.9]['Test Accuracy'].idxmax()
bayes = MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
print(models.iloc[best_index, :])


m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1']))

```
```python
Output:
 Predicted 0  Predicted 1
Actual 0         1435           18
Actual 1           15          204

```



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




