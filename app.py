from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os

app = Flask(__name__)


# Load the trained model and preprocessing objects
def load_model_and_encoders():
    # Load your data to get the column structure
    df = pd.read_csv('fertilizer_recommends.csv')

    # Initialize encoders and scaler
    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    fertilizer_encoder = LabelEncoder()
    scaler = StandardScaler()

    # Fit the encoders (same as in your notebook)
    df["Soil_Type"] = soil_encoder.fit_transform(df["Soil_Type"])
    df["Crop_Name"] = crop_encoder.fit_transform(df["Crop_Name"])
    df["Fertilizer_Name"] = fertilizer_encoder.fit_transform(df["Fertilizer_Name"])

    # Define features and target
    X = df.drop(columns=["Fertilizer_Name", "Crop_Name"])
    y = df[["Fertilizer_Name", "Crop_Name"]]

    # Fit the scaler
    X_scaled = scaler.fit_transform(X)

    # Train the model (or load if you have saved it)
    base_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    multi_output_model = MultiOutputClassifier(base_model)
    multi_output_model.fit(X_scaled, y)

    return {
        'model': multi_output_model,
        'soil_encoder': soil_encoder,
        'crop_encoder': crop_encoder,
        'fertilizer_encoder': fertilizer_encoder,
        'scaler': scaler,
        'feature_columns': X.columns
    }


# Load model and encoders when starting the app
model_data = load_model_and_encoders()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        pottasium = float(request.form['pottasium'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        ph = float(request.form['ph'])
        soil_type = request.form['soil_type']
        temperature = float(request.form['temperature'])

        # Encode soil type
        soil_encoded = model_data['soil_encoder'].transform([soil_type])[0]

        # Create input dictionary
        input_data = {
            "Nitrogen": nitrogen,
            "Phosphorous": phosphorous,
            "Pottasium": pottasium,
            "Humidity": humidity,
            "Rainfall": rainfall,
            "Ph": ph,
            "Soil_Type": soil_encoded,
            "Temperature": temperature
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=model_data['feature_columns'])

        # Scale the input
        input_scaled = model_data['scaler'].transform(input_df)

        # Make prediction
        predictions = model_data['model'].predict(input_scaled)[0]
        crop_pred = predictions[1]
        fertilizer_pred = predictions[0]

        # Decode predictions
        recommended_crop = model_data['crop_encoder'].inverse_transform([crop_pred])[0]
        recommended_fertilizer = model_data['fertilizer_encoder'].inverse_transform([fertilizer_pred])[0]

        return render_template('index.html',
                               recommendation=True,
                               crop=recommended_crop,
                               fertilizer=recommended_fertilizer,
                               form_data=request.form)

    # For GET request or initial load
    return render_template('index.html', recommendation=False)


if __name__ == '__main__':
    app.run(debug=True)