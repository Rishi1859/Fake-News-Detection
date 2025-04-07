from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask_cors import CORS

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def preprocess(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# API endpoint only
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news = data.get('news', '')
    if not news:
        return jsonify({'error': 'No news text provided'}), 400

    cleaned = preprocess(news)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = 'Fake' if prediction == 1 else 'Real'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
