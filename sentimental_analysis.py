import numpy as np
import pandas as pd
import nltk
import string
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle


def predict_sentiment(input_text):

    dataset = pd.DataFrame()
    dataset['comment'] = [input_text]

    print('Convert text to lower case')
    dataset['comment'] = dataset['comment'].str.lower()
    print(dataset.head())
    print('Remove Punctuation')
    print(string.punctuation)

    def remove_punc(text):
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        return text

    dataset['comment'] = dataset['comment'].apply(lambda x: remove_punc(x))
    print(dataset.head())
    print('Removing Numbers from the text')
    dataset['comment'] = dataset['comment'].str.replace('\d+', '')
    print(dataset.head())
    print('Tokenization of words')
    comment = dataset['comment']
    dataset['comment'] = comment.apply(word_tokenize)
    print(dataset.head())
    print('Removing stopwords')
    stopword = nltk.corpus.stopwords.words('english')
    print(stopword)

    def remove_stopwords(text):
        text = [word for word in text if word not in stopword]
        return text

    dataset['comment'] = dataset['comment'].apply(lambda x: remove_stopwords(x))
    print(dataset.head())
    print('Performing Stemming')
    st = nltk.PorterStemmer()

    def stemming(text):
        text = [st.stem(word) for word in text]
        return text

    dataset['comment'] = dataset['comment'].apply(lambda x: stemming(x))
    print(dataset.head())
    print('Performing Lemmatization')
    wn = nltk.WordNetLemmatizer()

    def lemmatizer(text):
        text = [wn.lemmatize(word) for word in text]
        return text


    dataset['comment'] = dataset['comment'].apply(lambda x: lemmatizer(x))
    print(dataset.head())

    print('Joining the words to form a string/sentence')
    corpus = []
    print(dataset.shape)
    data_length, _ = dataset.shape
    for i in range(0, data_length):
        review = ' '.join(dataset['comment'][i])
        corpus.append(review)

    print("Corpus", corpus)

    print(data_length)




    # Vectorizing the words and counting the frequency for each
    # Extracting the features
    f = open('my_classifier.pickle', 'rb')
    multinomial_NBC_classifier, cv = pickle.load(f)
    f.close() 

    X = cv.transform(corpus).toarray()
    print(X)
    print(len(X))
    # X= X.reshape(1, -1)

    
    y_pred_mnb = multinomial_NBC_classifier.predict(X[:])
    print("This is the prediction:", y_pred_mnb)
    return y_pred_mnb 