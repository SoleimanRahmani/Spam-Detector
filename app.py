from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


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
    #clf = MultinomialNB(alpha=0.110010) #Best classification model
    NB_spam_model = open('models/NB_spam_model.pkl', 'rb')
    clf = joblib.load(NB_spam_model)
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
