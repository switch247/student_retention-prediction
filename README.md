# Student Retention Prediction API

This is a Flask-based web application that uses a trained scikit-learn model to predict student retention based on various input features.

## Features

- Utilizes a RandomForestClassifier model to predict student retention.
- Provides a simple API endpoint at `/predict` to submit student data and receive a prediction.
- Handles JSON input data and returns JSON responses.
- Includes error handling for any exceptions that might occur during the prediction process.
- Maps the prediction numbers to meaningful labels: `0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'`.

## Requirements

- Python 3.x
- Flask
- Pandas
- Scikit-learn (sklearn)

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/student-retention-prediction.git
```

2. Change to the project directory:

```
cd student-retention-prediction
```

3. Create a virtual environment (optional, but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. Install the required dependencies:

```
pip install -r requirements.txt
```

5. Train the model and save it to a file (e.g., `model.pkl`). You can use the following example code to train the model:

```python
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
X_train, y_train = load_dataset()

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
pickle.dump(model, 'model.pkl')
```

6. Update the `'{path/to/model.pkl}'` in the Flask application code with the actual path to your saved model file.

## Usage

1. Start the Flask application:

```
python app.py
```

2. Send a POST request to the `/predict` endpoint with the student data in JSON format:

```json
{
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
```

it will automaticaly filter the feilds you need
which are

```
 ['Application mode', 'Displaced', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']

```

The API will respond with a JSON object containing the predicted student retention, which will be mapped to the corresponding label:

```json
{
  "prediction": "Enrolled"
}
```

## Contributing

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
