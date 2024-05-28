
import numpy as np
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import json

app = Flask(__name__)
# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)


# Define the feature column names
feature_columns = ['Application mode', 'Displaced', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()
        
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([data])
        
        # Select the relevant features
        X = input_data[feature_columns]
        # Save X as a JSON file
        # X.to_json('input_data.json', orient='records')
        # Make the prediction
        y_pred = model.predict(X)
        
        # Return the prediction as a JSON response
        # return jsonify({'prediction': y_pred[0].tolist()})
        dic = {0:'Dropout', 1:'Enrolled', 2:'Graduate'}


        # Map the numeric predictions to the corresponding dictionary values
        prediction = [dic[int(np.argmax(p))] for p in y_pred]

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})

    
    except Exception as e:
        # Handle any exceptions that might occur
        error_message = str(e)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)



# import requests
# import json

data = {
    "Marital status": 1,
    "Application mode": 8,
    "Application order": 5,
    "Course": 2,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 13,
    "Father's qualification": 10,
    "Mother's occupation": 6,
    "Father's occupation": 10,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 20,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 0,
    "Curricular units 1st sem (evaluations)": 0,
    "Curricular units 1st sem (approved)": 0,
    "Curricular units 1st sem (grade)": 0.0,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 0,
    "Curricular units 2nd sem (evaluations)": 0,
    "Curricular units 2nd sem (approved)": 0,
    "Curricular units 2nd sem (grade)": 0.0,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74
}

# response = requests.post('http://localhost:5000/predict', data=json.dumps(data), headers={'Content-Type': 'application/json'})
# print(response.json())