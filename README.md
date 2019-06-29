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
Output:

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
The best model is Naive Bayes with 98% accuracy. It classifies 91% of non-spam messages correctly (Model precision) and classifies the 93% of spam messages correctly (Model recall).

## Web Application

In the previous section, the code for classifying messages has been developed. In this section, a web application is developed that consists of a web page with a form field that let users enter a message. After submitting the message to the web application, it will render it on a new page which gives us a result of spam or not spam.

First, we create a folder for this project called ```Spam-Detector```. The folder is as follows:

```bash
spam.csv
app.py
check.py
templates/
        home.html
        result.html
static/
        style.css
```

The ```spam.csv``` is a collection of messages tagged as spam or ham. The ```templates``` is the directory in which Flask will look for static HTML files. The```home.html``` render an input text form where a user can enter a message and ```result.html``` shows the prediction based on the built classification model and user's input. ```style.css``` is saved in static folder which determine the look of HTML documents. ```app.py``` file contains the main machile learning code of the application that is executed by  Python interpreter to run the Flask web application. The ```check.py``` is an individual python file, added to the project directory just to demonstrate the model selection which is explained in the previous section.

### app.py
The app.py file is the main code executed by the Python interpreter to run the Flask web application. The code inside ```app.py``` is explained as follows:

#### Libraries
```python
from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
```
#### Flask
A Flask instance with the argument ```__name__``` is initialized to inform Flask about the HTML template folder (```templates```) which is in the same directory where it is located.



In the next step, the```@app.route('/')``` which is the route decorator is used to specify the URL that should trigger the execution of the ```home``` function. The ```home``` function renders the ```home.html``` HTML file, which is located in the ```templates``` folder.

```python
app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')
```

Inside the ```predict``` function, 
The access to spam data set, pre-processing the text, train-test spliting, decision making (prediction) and storing the model are achieved inside the ```predict``` function. The input message, entered by the user is also used in this part to make a prediction for its label.

```python
@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Naive Bayes Classifier
    clf = MultinomialNB(alpha=0.110010) #Best classification model
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prediction_score = round(accuracy_score(y_test, pred) * 100, 2)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('result.html', prediction=my_prediction, prediction_score=prediction_score)

if __name__ == '__main__':
    app.run(debug=True)
```

The ```POST``` method is utilized in order to transport the form data to the server in the message body. By setting the ```debug=True``` argument inside the ```app.run``` , Flask's debugger is activated. FInally, the function ```run```is used to run the application on the server while the ```if``` statement of ```__name__ == '__main__'``` is true which shows that script is directly executed by the Python interpreter.


### home.html
The home.html file renders a text form where a user can enter a message.
```python
<!DOCTYPE html>
<html>
<head>
	<title>Home</title>
	<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/styles.css') }}">
</head>
<body>
	<header>
		<div class="container">
		<div id="brandname">
			<h1  style="color:blue;" align="center">Spam Detector</h1>
		</div>
		<h2 style="color:black;" align="center">Machine Learning App with Flask API to submit data for classification</h2>
	</div >
	</header>
	<div class="ml-container" style="background-color:lightblue" align="center">
		<form action="{{ url_for('predict')}}" method="POST">
		<p>Enter Your Message Here Please:</p>
		<textarea name="message" rows="4" cols="50"></textarea>
		<br/>
		<button type="submit" class="btn btn-primary">Predict</button>
	</form>
	</div>
</body>
</html>
```

The ```home.html``` is used in ```app.py``` according to the following code:

```python
@app.route('/')
def home():
	return render_template('home.html')
```

### result.HTML
The ```result.html```file is the page where the prediction of the user's input will be shown. The ```result.html``` contains the following content:
```python
<!DOCTYPE html>
<html>
<head>
	<title></title>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/styles.css') }}">
</head>
<body>
	<header>
		<div class="container">
		<div id="brandname">
			<h1  style="color:blue;" align="center">Spam Detector</h1>
		</div>
		<h2 style="color:black;" align="center">Machine Learning App with Flask API to submit data for classification</h2>
	    </div>
	</header>
	<p style=color:"blue" font-size=20 text-align="center" align="center"><b>Result of prediction:</b></p>
	<div class="results"  style="background-color:lightblue" align="center">
	{% if prediction == 1%}
	<h2 style="color:red;" align="center">Message is Spam!</h2>
	{% elif prediction == 0%}
	<h2 style="color:blue;" align="center">This Message is NOT Spam</h2>
	{% endif %}
	</div>
	<div class="alert alert-info" role="alert" align="center" style="background-color:lightblue">
		<p><span style="color:red"  type=number step=0.01>Prediction Score is:</span>: {{ prediction_score }} %</p>
	</div>
</body>
</html>
```

The ```result.html``` file is defined in the ```app.py``` script and is rendered via ```the render_template('result.html', prediction=my_prediction)``` line which is returned inside the ```predict``` function. The ```{% if prediction ==1%},{% elif prediction == 0%},{% endif %}```in ```result.htm``` script is used to access the prediction returned from our HTTP request within the HTML file.

### styles.css

The ```styles.css``` file is used in the header section of ```home.html``` and ```result.html``` which determines the shape of these HTML documents. The ```styles.css``` contains the following content:

```python
body{
	font:15px/1.5 Arial, Helvetica,sans-serif;
	padding: 0px;
	background-color:Blue;
}
.container{
	width:100%;
	margin: auto;
	overflow: hidden;
}

header{
	background:#03A9F4;#35434a;
	border-bottom:#448AFF 3px solid;
	height:120px;
	width:100%;
	padding-top:30px;
}
.main-header{
			text-align:center;
			background-color: blue;
			height:100px;
			width:100%;
			margin:0px;
		}
#brandname{
	float:left;
	font-size:300px;
	color: #fff;
	margin: 10px;
}
header h2{
	text-align:center;
	color:#fff;
}
.btn-info {background-color: #2196F3;
	height:40px;
	width:100px;} /* Blue */
.btn-info:hover {background: #0b7dda;}


.resultss{
	border-radius: 15px 50px;
    background: #345fe4;
    padding: 20px; 
    width: 200px;
    height: 150px;
}
```
## Web Page

All the files in the directory is explained above. THe next step is to runthe API by executing the following command in the Terminal:

```
cd Spam-Detector
python app.py
```
And the output would be:
```
UIT-VPS-26NBR:Spam-Detector srahmani$ python3 app.py
  import imp
 * Serving Flask app "app"
 * Environment: production
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Debugger is active!
 * Debugger PIN: 126-674-093
```

Now by openning the web browser and navigating to http://127.0.0.1:5000/, a  website with the following content will come up:

<img width="824" alt="Screen Shot 2019-06-26 at 8 35 22 PM" src="https://user-images.githubusercontent.com/45254300/60224871-5a39fa00-9852-11e9-96ac-33d915bdd4a0.png">


As it can be seen, the text can be written in the box. By clicking ```Predict``` botton, the prediction will show up. For instance, by writing the following message in the box:

<img width="802" alt="Screen Shot 2019-06-29 at 7 14 48 PM" src="https://user-images.githubusercontent.com/45254300/60390398-b9427d80-9aa3-11e9-9a2a-8a845e2b09dd.png">

And clicking the ```Predict``` botton, the following prediction will come up:

<img width="838" alt="Screen Shot 2019-06-29 at 7 26 58 PM" src="https://user-images.githubusercontent.com/45254300/60390405-ebec7600-9aa3-11e9-8f66-221402b479ec.png">

Which shows that the model predicttion is correct and the message is NOT spam.
As another example, by writing the following message in the box:


And clicking the ```Predict``` botton, the following prediction will come up:


Which again shows that the model predicttion is correct and in this time, the message is SPAM.



