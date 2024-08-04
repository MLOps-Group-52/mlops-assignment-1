


# The code snippet importing necessary modules and libraries 
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
# `from flasgger import Swagger` is importing the Swagger module from the Flasgger library. Swagger is
# a tool that helps in documenting and visualizing APIs. In this code snippet, it is being used to
# generate API documentation for the Flask application.
from flasgger import Swagger

# `app = Flask(__name__)` is creating a Flask application instance. 
app = Flask(__name__)

# `swagger = Swagger(app)` is initializing the Swagger module with the Flask application instance
# `app`. This line of code sets up Swagger to generate API documentation for the Flask application.
swagger = Swagger(app)

# `CORS(app)` is enabling Cross-Origin Resource Sharing (CORS) for the Flask application instance
CORS(app)

# The `add_relationship` function in this Python code snippet adds a relationship between two entities
# to a knowledge graph.
# :return: The `add_relationship` function returns a JSON response with a success message if the
# relationship between Entity1 and Entity2 is added successfully to the knowledge graph. If there are
# any issues such as an empty request body or invalid data, appropriate error messages are returned
# with status codes 400 or 500.
@app.route('/predict', methods=['POST'])
def predict():
    
    #swagger documentation start
    """
    Adds the  Relationship between Entity1 and Entity2  to the knowledge Graph.
    ---
    parameters:
    -   name: body
        in: body
        required: true
        schema:
            type: object
            properties:
                entity1:
                    type: string
                    example: Advocate John
                entity2:
                    type: string
                    example: Case Number 147/2023
                Relationship:
                    type: string
                    example: Filed
    responses:
        200:
            description: Returns a successful message.
        400:
            description: Bad Request if data is empty or invalid
        500:
            description: Internal issue.
    """
    #swagger documentation end
    try:
    # This of code in the Flask application that is extracting JSON data
    # from the incoming request. When a client sends a POST request to the `/addRelation`,
    # endpoints with JSON data in the request body, `request.json` is
    # used to access this JSON data within the Flask route function.
        data = request.json
        
        # Data validation
        if not data:
            return jsonify(error="Empty request body"), 400
    
    # The exception handling to handle the exception.
    except Exception as e:
        return jsonify(error="Internal Exception Occurred"), 500
    
    # Return successful message with status code 200
    return jsonify({'message': 'Relationship added successfully'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
