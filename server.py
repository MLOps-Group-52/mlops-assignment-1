# Import necessary modules and libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from flasgger import Swagger

# Create a Flask application instance
app = Flask(__name__)

# Initialize Swagger for API documentation
swagger = Swagger(app)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Load the trained model
model = joblib.load('Models/model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the Taxi fare.
    ---
    parameters:
    - name: body
      in: body
      required: true
      schema:
        type: object
        properties:
          trip_duration:
            type: number
            example: 2
          distance_traveled:
            type: number
            example: 10
          num_of_passengers:
            type: number
            example: 1
          tip:
            type: number
            example: 20.2
          miscellaneous_fees:
            type: number
            example: 23.2
          fare:
            type: number
            example: 20.2
          surge_applied:
            type: boolean
            example: true
    responses:
      200:
        description: Returns the predicted fare.
      400:
        description: Bad request if data is empty or invalid.
      500:
        description: Internal server error.
    """
    try:
        # Extract JSON data from the request
        data = request.json

        # Validate data
        if not data:
            return jsonify(error="Empty request body"), 400

        # Extract features from request
        features = [
            data.get('trip_duration'),
            data.get('distance_traveled'),
            data.get('fare'),
            data.get('tip'),
            data.get('miscellaneous_fees'),
            data.get('num_of_passengers'),
            data.get('surge_applied')
        ]

        # Check if all required features are provided
        if None in features:
            return jsonify(error="Missing data in request"), 400

        # Convert features to numpy array for prediction
        features_array = np.array([features])

        # Make prediction
        prediction = model.predict(features_array)[0]
        return jsonify(prediction=prediction)

    except Exception as e:
        return jsonify(error=f"Internal Exception Occurred: {str(e)}"), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
