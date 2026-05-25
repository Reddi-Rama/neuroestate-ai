from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# LOAD MODEL
model = joblib.load("../models/house_price_model.pkl")


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        bedrooms = int(data["Bedrooms"])

        bathrooms = int(data["Bathrooms"])

        total_rooms = bedrooms + bathrooms

        input_data = pd.DataFrame({

            "SqFt": [int(data["SqFt"])],

            "Bedrooms": [bedrooms],

            "Bathrooms": [bathrooms],

            "Brick": [data["Brick"]],

            "Neighbourhood": [data["Neighbourhood"]],

            "TotalRooms": [total_rooms]

        })

        prediction = model.predict(input_data)

        return jsonify({

            "Predicted_Price": float(prediction[0])

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        })


if __name__ == "__main__":

    app.run(debug=True)