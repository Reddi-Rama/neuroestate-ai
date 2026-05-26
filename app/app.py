from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("models/house_price_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction_text = ""

    if request.method == "POST":
        try:
            sqft = float(request.form["sqft"])
            bedrooms = int(request.form["bedrooms"])
            bathrooms = int(request.form["bathrooms"])
            brick = request.form["brick"]
            neighbourhood = request.form["neighbourhood"]

            # Create TotalRooms
            total_rooms = bedrooms + bathrooms

            # Create dataframe
            input_data = pd.DataFrame({
                "SqFt": [sqft],
                "Bedrooms": [bedrooms],
                "Bathrooms": [bathrooms],
                "Brick": [brick],
                "Neighbourhood": [neighbourhood],
                "TotalRooms": [total_rooms]
            })

            # Predict
            prediction = model.predict(input_data)[0]

            prediction_text = f"Estimated Price: ${round(prediction, 2)}"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction_text=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)