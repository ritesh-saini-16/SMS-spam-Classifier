
import pandas as pd
import string
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

nltk.download('punkt')
nltk.download('stopwords')

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

# Read data
try:
    df = pd.read_csv('spam.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['v1', 'v2']].dropna()
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['transformed_text'] = df['text'].apply(transform_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['transformed_text'])
y = df['label'].values

# Train model
model = MultinomialNB()
model.fit(X, y)

# Convert model to ONNX
initial_type = [('float_input', FloatTensorType([None, 3000]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# Save vectorizer data (vocab and idf)
vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
idf = vectorizer.idf_.tolist()

with open('vectorizer_data.json', 'w') as f:
    json.dump({
        'vocab': vocab,
        'idf': idf,
        'max_features': 3000
    }, f)

print('Model and vectorizer data exported successfully!')
