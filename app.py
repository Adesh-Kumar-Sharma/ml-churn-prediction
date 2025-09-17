from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import os

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run before the app starts accepting requests
    print("Loading ML model...")
    ml_models["churn_model"] = joblib.load("model/churn_model.pkl")
    print("Model loaded successfully!")
    yield
    # Code to run when the app is shutting down
    print("Clearing ML models...")
    ml_models.clear()

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Define the list of origins (Next.js) that are allowed to make requests
origins = [
    "http://localhost:3000",
    "https://ml-churn-frontend.vercel.app"
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load model and encoders
try:
    model = joblib.load('model/churn_model.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
except:
    model = None
    label_encoders = {}
    feature_names = []

# Pydantic models for request/response
class CustomerData(BaseModel):
    tenure: int = 12
    MonthlyCharges: float = 65.0
    TotalCharges: float = 780.0
    Contract: str = "Month-to-month"
    PaymentMethod: str = "Electronic check"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    TechSupport: str = "No"

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str

# Routes
@app.get("/")
async def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": "loaded" if model else "not_loaded"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    # Get the model from the dictionary
    model = ml_models["churn_model"]

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dictionary
        customer_dict = customer.dict()
        
        # Create DataFrame
        df = pd.DataFrame([customer_dict])
        
        # Apply label encoding to categorical features
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select features in correct order
        X = df[feature_names]
        
        # Make prediction
        prediction_proba = model.predict_proba(X)
        prediction = model.predict(X)

        # Select the probability for the "Will Churn" class
        churn_probability = float(prediction_proba[0][1])
        
        # Get the prediction result (0 or 1) from the array
        prediction_result = prediction[0]
        
        return PredictionResponse(
            churn_probability=churn_probability,
            prediction="Will Churn" if prediction_result == 1 else "Will Not Churn",
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": feature_names,
        "n_features": len(feature_names)
    }

# Train endpoint (for demonstration)
@app.post("/retrain")
async def retrain_model():
    try:
        from train_model import train_model
        accuracy = train_model()
        
        # Reload model
        global model, label_encoders, feature_names
        model = joblib.load('model/churn_model.pkl')
        label_encoders = joblib.load('model/label_encoders.pkl')
        feature_names = joblib.load('model/feature_names.pkl')
        
        return {
            "message": "Model retrained successfully",
            "accuracy": accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
