from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model
model_path = os.path.join("models", "house_price_model.pkl")
model = joblib.load(model_path)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        sqft = float(request.form['sqft'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        brick = request.form['brick']
        neighbourhood = request.form['neighbourhood']

        # Create TotalRooms
        total_rooms = bedrooms + bathrooms

        # Input dataframe
        input_data = pd.DataFrame({
            'SqFt': [sqft],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Brick': [brick],
            'Neighbourhood': [neighbourhood],
            'TotalRooms': [total_rooms]
        })

        # Predict
        prediction = model.predict(input_data)[0]

        prediction = round(prediction, 2)

        return render_template(
            'index.html',
            prediction_text=f"Estimated Price: ${prediction}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)