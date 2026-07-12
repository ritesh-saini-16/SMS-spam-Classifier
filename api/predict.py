
from flask import Flask, request, jsonify
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
vectorizer = pickle.load(open(os.path.join(os.path.dirname(__file__), 'vectorizer.pkl'), 'rb'))
model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'model.pkl'), 'rb'))

app = Flask(__name__)

@app.route('/', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_sms = data.get('message', '')
    if not input_sms:
        return jsonify({'error': 'No message provided'}), 400
    
    transformed_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    return jsonify({'prediction': int(result)})

# For Vercel
handler = app.wsgi_app
