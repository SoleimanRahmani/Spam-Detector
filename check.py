import xgboost as xgboost
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import tree
from mleap.sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



df = pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)  # Fit the Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes Classifier

clf = MultinomialNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
prediction_score = round(clf.score(X_test, y_test) * 100, 2)

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

model3 = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
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

for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy of %s is %s" % (clf, acc))
    print("--------------------------------------------------------------------------")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of %s is %s" % (clf, cm))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


