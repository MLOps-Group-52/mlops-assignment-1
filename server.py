from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load('Models/model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Assuming the input is a JSON with the features
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)