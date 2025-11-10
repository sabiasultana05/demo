import os
import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# create the Flask app
app = Flask(__name__)

# load model using a path relative to this file
MODEL_FILENAME = 'House Price Prediction Model.pkl'
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    # templates folder contains `home.html` so render that
    return render_template('index.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    # get JSON body
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No JSON body provided'}), 400

    # support either {"data": {...}} or direct feature dict/list
    if isinstance(data, dict) and 'data' in data:
        data = data['data']

    # build numpy array for prediction
    try:
        if isinstance(data, dict):
            arr = np.array(list(data.values())).reshape(1, -1)
        else:
            arr = np.array(data).reshape(1, -1)
    except Exception as e:
        return jsonify({'error': 'Invalid input format', 'details': str(e)}), 400

    prediction = model.predict(arr)
    output = float(prediction[0]) if hasattr(prediction[0], '__float__') else prediction[0]
    return jsonify({'prediction': output})

@app.route('/predict', methods=['POST'])
def predict():
    # Read form values and convert to float
    try:
        data = [float(x) for x in request.form.values()]
    except Exception:
        return render_template('index.html', prediction_text='Invalid input: please enter numeric values')

    # build a 2D numpy array with shape (1, n_features)
    final_input = np.array(data).reshape(1, -1)
    try:
        output = model.predict(final_input)[0]
    except Exception as e:
        return render_template('index.html', prediction_text=f'Prediction error: {e}')

    return render_template('index.html', prediction_text=f'Predicted House Price is {output}')


if __name__ == '__main__':
    app.run(debug=True)
