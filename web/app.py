from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.transport_delay.cleaning import _normalize_route_id, _normalize_weather
from src.transport_delay.features import add_features


class TripInput(BaseModel):
    route_id: str
    scheduled_time: str
    weather: str
    passenger_count: int
    latitude: float
    longitude: float


app = FastAPI(title="Bus Delay Predictor", description="Predict bus delays using ML model")

# Load the trained model and preprocessing pipeline
MODEL_PATH = Path(__file__).parent.parent / "outputs" / "random_forest_model.pkl"
PREPROCESSOR_PATH = Path(__file__).parent.parent / "outputs" / "preprocessor.pkl"

model = None
preprocessor = None
feature_names = []


def load_model():
    global model, preprocessor, feature_names
    if model is None:
        # For now, we'll train and save the model on first load
        from src.transport_delay.pipeline import run_pipeline
        
        project_dir = Path(__file__).parent.parent
        input_csv = project_dir / "dirty_transport_dataset.csv"
        output_dir = project_dir / "outputs"
        
        # Run pipeline to get trained model
        out = run_pipeline(input_csv=input_csv, output_dir=output_dir)
        
        # Extract the trained Random Forest model
        from src.transport_delay.modeling import train_and_evaluate
        categorical = ["route_id", "weather", "time_of_day", "day_type"]
        numeric = ["scheduled_hour", "scheduled_dayofweek", "passenger_count", "latitude", "longitude", "weather_severity", "route_frequency"]
        
        df = out.cleaned
        X = df[categorical + numeric].copy()
        y = df["delay_minutes"].astype(float)
        
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
        
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", Pipeline([("scaler", StandardScaler())]), numeric),
            ],
            remainder="drop",
        )
        
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        pipe = Pipeline([("pre", pre), ("model", rf)])
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)
        
        model = pipe
        preprocessor = pre
        
        # Get feature names after one-hot encoding
        fitted_pre = pipe.named_steps["pre"]
        feature_names = []
        for name, transformer, cols in fitted_pre.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(list(transformer.get_feature_names_out(cols)))
            else:
                feature_names.extend(list(cols))


def preprocess_input(trip: TripInput) -> pd.DataFrame:
    """Convert user input to model-ready format"""
    # Normalize inputs
    route_norm = _normalize_route_id(trip.route_id) or "R00"
    weather_norm = _normalize_weather(trip.weather) or "sunny"
    
    # Create base record
    record = {
        "route_id": route_norm,
        "scheduled_time": trip.scheduled_time,
        "actual_time": "",  # We'll predict this
        "weather": weather_norm,
        "passenger_count": trip.passenger_count,
        "latitude": trip.latitude,
        "longitude": trip.longitude,
    }
    
    df = pd.DataFrame([record])
    
    # Add features
    df = add_features(df)
    
    return df


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the prediction interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bus Delay Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 4px; }
            .error { color: #dc3545; }
            .success { color: #28a745; }
        </style>
    </head>
    <body>
        <h1>Bus Delay Predictor</h1>
        <p>Enter trip details to predict expected delay:</p>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="route_id">Route ID:</label>
                <input type="text" id="route_id" name="route_id" placeholder="e.g., R01, Route-2, 3" required>
            </div>
            
            <div class="form-group">
                <label for="scheduled_time">Scheduled Time:</label>
                <input type="datetime-local" id="scheduled_time" name="scheduled_time" required>
            </div>
            
            <div class="form-group">
                <label for="weather">Weather:</label>
                <select id="weather" name="weather" required>
                    <option value="sunny">Sunny</option>
                    <option value="cloudy">Cloudy</option>
                    <option value="rainy">Rainy</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="passenger_count">Passenger Count:</label>
                <input type="number" id="passenger_count" name="passenger_count" min="1" max="200" placeholder="1-200" required>
            </div>
            
            <div class="form-group">
                <label for="latitude">Latitude:</label>
                <input type="number" id="latitude" name="latitude" step="0.000001" placeholder="e.g., 24.7128" required>
            </div>
            
            <div class="form-group">
                <label for="longitude">Longitude:</label>
                <input type="number" id="longitude" name="longitude" step="0.000001" placeholder="e.g., 46.6753" required>
            </div>
            
            <button type="submit">Predict Delay</button>
        </form>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = {
                    route_id: formData.get('route_id'),
                    scheduled_time: formData.get('scheduled_time').replace('T', ' '),
                    weather: formData.get('weather'),
                    passenger_count: parseInt(formData.get('passenger_count')),
                    latitude: parseFloat(formData.get('latitude')),
                    longitude: parseFloat(formData.get('longitude'))
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data),
                    });
                    
                    const result = await response.json();
                    
                    const resultDiv = document.getElementById('result');
                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <h3 class="success">Prediction Result</h3>
                            <p><strong>Predicted Delay:</strong> ${result.predicted_delay.toFixed(2)} minutes</p>
                            <p><strong>Confidence:</strong> ${result.confidence}</p>
                            <h4>Top Influencing Factors:</h4>
                            <ul>
                                ${result.top_factors.map(f => `<li>${f.feature}: ${f.importance.toFixed(3)}</li>`).join('')}
                            </ul>
                        `;
                    } else {
                        resultDiv.innerHTML = `<h3 class="error">Error</h3><p>${result.detail}</p>`;
                    }
                    resultDiv.style.display = 'block';
                } catch (error) {
                    document.getElementById('result').innerHTML = `<h3 class="error">Network Error</h3><p>${error.message}</p>`;
                    document.getElementById('result').style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def predict(trip: TripInput):
    """Make prediction for a single trip"""
    try:
        # Preprocess input
        df = preprocess_input(trip)
        
        # Extract features for model
        categorical = ["route_id", "weather", "time_of_day", "day_type"]
        numeric = ["scheduled_hour", "scheduled_dayofweek", "passenger_count", "latitude", "longitude", "weather_severity", "route_frequency"]
        
        X = df[categorical + numeric].copy()
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get feature importance
        importances = model.named_steps["model"].feature_importances_
        feature_imp = list(zip(feature_names, importances))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        top_factors = [
            {"feature": name, "importance": imp} 
            for name, imp in feature_imp[:5]
        ]
        
        # Simple confidence based on prediction range
        if abs(prediction) < 10:
            confidence = "High"
        elif abs(prediction) < 30:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "predicted_delay": float(prediction),
            "confidence": confidence,
            "top_factors": top_factors
        }
        
    except Exception as e:
        return {"detail": f"Prediction error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
