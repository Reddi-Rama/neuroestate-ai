from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/house_price_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction_text = None

    if request.method == "POST":

        try:

            sqft = float(request.form.get("sqft"))
            bedrooms = int(request.form.get("bedrooms"))
            bathrooms = int(request.form.get("bathrooms"))
            brick = request.form.get("brick")
            neighbourhood = request.form.get("neighbourhood")

            total_rooms = bedrooms + bathrooms

            input_data = pd.DataFrame({
                "SqFt": [sqft],
                "Bedrooms": [bedrooms],
                "Bathrooms": [bathrooms],
                "Brick": [brick],
                "Neighbourhood": [neighbourhood],
                "TotalRooms": [total_rooms]
            })

            prediction = model.predict(input_data)

            prediction_text = f"Estimated Price: ${round(prediction[0], 2)}"

        except Exception as e:

            prediction_text = f"Error: {e}"

    return render_template(
        "index.html",
        prediction_text=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)