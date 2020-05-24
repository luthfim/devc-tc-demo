import os

from flask import Flask, jsonify, request
from sklearn.externals import joblib
from utils import strip_html, remove_special_characters
from sklearn.feature_extraction.text import CountVectorizer

# define paths
output_dir = '../output'
clf_name = 'clf.joblib'
vec_name = 'vec.joblib'
clf_path = os.path.join(output_dir, clf_name)
vec_path = os.path.join(output_dir, vec_name)

# load bin files
clf = joblib.load(clf_path)
vectorizer = joblib.load(vec_path)

# run server

app = Flask(__name__)

def preprocess(text):
    text = strip_html(text)
    text = remove_special_characters(text)
    text = text.lower()
    return text

def postprocess(pred):
    class_map = {0: 'Not spam', 1: 'Spam'}
    return [class_map[p] for p in pred]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    text = data.get('text', [])
    text = [preprocess(t) for t in text]
    bow = vectorizer.transform(text)

    pred = clf.predict(bow)
    pred = postprocess(pred)

    rsp = {'class': pred}
    return jsonify(rsp)

if __name__ == '__main__':
    app.run(port=8080)
