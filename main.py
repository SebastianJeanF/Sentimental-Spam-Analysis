from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk
import sentimental_analysis 
import pickle
import pandas as pd
import spam_detection

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')
  nltk.download('wordnet')
  nltk.download('stopwords')







set(stopwords.words('english'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/detect')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def post_review():
    stop_words = stopwords.words('english')
    input_text= request.form['input_text']

    prediction_result = sentimental_analysis.predict_sentiment(input_text)
    # processed_doc1 = ' '.join([word for word in input_text.split() if word not in stop_words])

    # sia = SentimentIntensityAnalyzer()
    # ps = sia.polarity_scores(text=input_text)
    # result = round((1 + ps['compound'])/2, 2) * 100
    
    return render_template('home2.html', predict=prediction_result , text=input_text)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    my_prediction = spam_detection.predict_spam(message)
    
    return render_template('index.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1:5000", threaded=True)
