from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and columns
def load_model():
    with open("credit_score_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, columns

model, model_columns = load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []
    for col in model_columns:
        val = request.form.get(col)
        print(f"Received {col}: {val}")  # Log the input data received
        try:
            input_data.append(float(val))
        except (TypeError, ValueError):
            input_data.append(0.0)  # Default value if invalid input

    # Log the entire input data
    print("Input Data:", input_data)

    # Convert the input data to a DataFrame to match the model's expected input format
    input_df = pd.DataFrame([input_data], columns=model_columns)

    # Log the DataFrame to see if the conversion was correct
    print("Input DataFrame:")
    print(input_df)

    try:
        # Predict using the model
        prediction = model.predict(input_df)[0]
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return render_template("index.html", prediction_text="Error in prediction", prediction_score=0)

    # Map the model's prediction to a label
    label_map = {0: "Bad", 1: "Good", 2: "Standard"}
    result = label_map.get(prediction, "Unknown")

    # Map prediction to a score (e.g., 0 = Bad, 1 = Good, 2 = Standard)
    prediction_score = (prediction + 1) * 33.33  # Scale the result to a 0-100 scale

    # Log the result and return to the template
    print(f"Prediction Result: {result}, Score: {prediction_score}")
    
    return render_template("index.html", prediction_text=result, prediction_score=prediction_score)


if __name__ == "__main__":
    app.run(debug=False)
