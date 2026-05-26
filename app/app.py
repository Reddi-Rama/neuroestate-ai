from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/house_price_model.pkl", "rb"))

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
            currency = request.form["currency"]

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
            prediction = model.predict(input_data)

            predicted_price = float(prediction[0])

            # Currency conversion
            if currency == "INR":
                predicted_price *= 83
                symbol = "₹"

            elif currency == "EUR":
                predicted_price *= 0.92
                symbol = "€"

            elif currency == "GBP":
                predicted_price *= 0.79
                symbol = "£"

            elif currency == "JPY":
                predicted_price *= 156
                symbol = "¥"

            else:
                symbol = "$"

            prediction_text = f"Estimated Price: {symbol} {round(predicted_price, 2)}"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)