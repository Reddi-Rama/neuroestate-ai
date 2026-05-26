from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        sqft = int(request.form["sqft"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        brick = request.form["brick"]
        neighbourhood = request.form["neighbourhood"]

        total_rooms = bedrooms + bathrooms

        input_data = pd.DataFrame({
            "SqFt": [sqft],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Brick": [brick],
            "Neighbourhood": [neighbourhood],
            "TotalRooms": [total_rooms]
        })

        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)