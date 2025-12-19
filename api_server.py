from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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


app = FastAPI(title="Bus Delay Predictor API", description="API for predicting bus delays")

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and preprocessing
model = None
preprocessor = None
feature_names = []
route_frequencies = {}


def load_model():
    """Load the trained model and preprocessing pipeline"""
    global model, preprocessor, feature_names, route_frequencies
    if model is None:
        # First, run the pipeline to get the properly trained model
        from src.transport_delay.pipeline import run_pipeline
        
        project_dir = Path(__file__).resolve().parent
        input_csv = project_dir / "dirty_transport_dataset.csv"
        output_dir = project_dir / "outputs"
        
        # Run pipeline to get trained model
        out = run_pipeline(input_csv=input_csv, output_dir=output_dir)
        
        # Load the cleaned data and train the model properly
        from src.transport_delay.modeling import train_and_evaluate
        categorical = ["route_id", "weather", "time_of_day", "day_type"]
        numeric = ["scheduled_hour", "scheduled_dayofweek", "passenger_count", "latitude", "longitude", "weather_severity", "route_frequency"]
        
        df = out.cleaned
        X = df[categorical + numeric].copy()
        y = df["delay_minutes"].astype(float)
        
        # Store route frequencies from training data
        route_frequencies = df.groupby("route_id")["route_id"].count().to_dict()
        
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
        
        # Train on ALL data for maximum accuracy
        pipe.fit(X, y)
        
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
        
        print(f"Model loaded with {len(feature_names)} features")


def validate_input(trip: TripInput) -> tuple[bool, str]:
    """Validate input data and return (is_valid, error_message)"""
    
    # Validate passenger count
    if trip.passenger_count < 1 or trip.passenger_count > 200:
        return False, f"Invalid passenger count: {trip.passenger_count}. Must be between 1 and 200."
    
    # Validate GPS coordinates
    if abs(trip.latitude) > 90:
        return False, f"Invalid latitude: {trip.latitude}. Must be between -90 and 90."
    
    if abs(trip.longitude) > 180:
        return False, f"Invalid longitude: {trip.longitude}. Must be between -180 and 180."
    
    # Validate route ID format
    if not trip.route_id or len(trip.route_id.strip()) == 0:
        return False, "Route ID cannot be empty."
    
    # Validate weather
    valid_weather = ["sunny", "cloudy", "rainy"]
    if trip.weather.lower() not in valid_weather:
        return False, f"Invalid weather: {trip.weather}. Must be one of: {valid_weather}."
    
    # Validate scheduled time format
    try:
        pd.to_datetime(trip.scheduled_time)
    except Exception:
        return False, f"Invalid scheduled time format: {trip.scheduled_time}. Use format like '2025-01-20 08:30'."
    
    return True, ""


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
    
    # Convert time strings to datetime
    df["scheduled_dt"] = pd.to_datetime(df["scheduled_time"], errors="coerce")
    df["actual_dt"] = pd.NaT  # Not a Time - we're predicting this
    
    # Manually add features without delay calculation
    out = df.copy()
    
    out["scheduled_hour"] = out["scheduled_dt"].dt.hour
    out["scheduled_dayofweek"] = out["scheduled_dt"].dt.dayofweek
    out["day_type"] = np.where(out["scheduled_dayofweek"] >= 5, "weekend", "weekday")

    def _tod(h: float) -> str:
        if 5 <= h <= 11:
            return "morning"
        if 12 <= h <= 16:
            return "afternoon"
        if 17 <= h <= 21:
            return "evening"
        return "night"

    out["time_of_day"] = out["scheduled_hour"].apply(_tod)

    severity_map = {"sunny": 1, "cloudy": 2, "rainy": 3}
    out["weather_severity"] = out["weather"].map(severity_map).fillna(2).astype(int)

    # Use route frequency from training data, not from single prediction
    out["route_frequency"] = out["route_id"].map(route_frequencies).fillna(1).astype(int)
    
    # Add placeholder delay (will be predicted)
    out["delay_minutes"] = 0.0  # Placeholder, won't be used for prediction
    
    return out


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/")
async def root():
    """API health check"""
    return {"message": "Bus Delay Predictor API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(feature_names) if feature_names else 0
    }


@app.post("/predict")
async def predict(trip: TripInput):
    """Make prediction for a single trip"""
    try:
        # Validate input first
        is_valid, error_msg = validate_input(trip)
        if not is_valid:
            return {
                "status": "error",
                "detail": error_msg
            }
        
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
            "top_factors": top_factors,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "detail": f"Prediction error: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
