from flask import Flask, render_template, request, Response
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import pandas as pd
from nltk.corpus import stopwords

app = Flask(__name__)
loaded_model = load_model('LSTM.h5')
english_stops = set(stopwords.words('english'))
max_length=130


def load_dataset():
    df = pd.read_csv('product.csv')
    x_data = df['review']  # Reviews/Input
    y_data = df['sentiment']  # Sentiment/Output

    # PRE-PROCESS REVIEW
    x_data = x_data.replace({'<.*?>': ''}, regex=True)  # remove html tag
    x_data = x_data.replace({'[^A-Za-z]': ' '}, regex=True)  # remove non alphabet
    x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops])  # remove stop words
    x_data = x_data.apply(lambda review: [w.lower() for w in review])  # lower case

    # ENCODE SENTIMENT -> 0 & 1
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    return x_data, y_data
x_data, y_data = load_dataset()




@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')




@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        review= request.form['review']
        token = Tokenizer(lower=False)
        token.fit_on_texts(x_data)
        regex = re.compile(r'[^a-zA-Z\s]')
        review = regex.sub('', review)
        words = review.split(' ')
        filtered = [w for w in words if w not in english_stops]
        filtered = ' '.join(filtered)
        filtered = [filtered.lower()]
        tokenize_words = token.texts_to_sequences(filtered)
        tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
        result = loaded_model.predict(tokenize_words)
        print(result)
        return render_template('result.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)






