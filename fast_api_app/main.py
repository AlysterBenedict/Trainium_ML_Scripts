# =====================================================================================
# FastAPI Server Entrypoint File
# =====================================================================================
# USE CASE:
# Acts as the production REST API gateway for the AI Gym ecosystem. It initializes 
# pre-trained neural networks and preprocessors, defines Pydantic payload models, 
# and routes external client HTTP requests for biometric predictions and plan generation.
#
# ENDPOINTS DEFINED:
# - GET  /                  : Server health check / greeting.
# - POST /predict_biometrics: Receives front/side photos, extracts silhouettes, and returns estimated dimensions.
# - POST /generate_workout  : Receives biometric details and targets, and autoregressively generates a 30-day program.
#
# SUB-MODULES & ARTIFACTS CALLED BY THIS FILE:
# 1. recommendation.biometric_estimator.BiometricEstimator: Used to process raw user photos and
#    estimate 15 key body measurements (via 'best_bodym_model.pth').
# 2. recommendation.workout_generator.WorkoutGenerator: Used to ingest biometrics and autoregressively
#    generate the progressive 30-day workout plan sequence.
# 3. best_bodym_model.pth: Loaded via BiometricEstimator to supply neural weights for silhouette regression.
# 4. trainium_sota_transformer_model.pth: Loaded via WorkoutGenerator to supply neural weights for generation.
# 5. tokenizer.json: Loaded via WorkoutGenerator to decode forecasted tokens into exercise names.
# 6. scaler.pkl: Loaded via WorkoutGenerator to scale incoming profile values (StandardScaler).
# 7. encoder.pkl: Loaded via WorkoutGenerator to encode categorical goals/levels (OneHotEncoder).
#
# CALLED / IMPORTED BY:
# - This file is executed directly as the server entrypoint.
# - Command: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# =====================================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from recommendation.biometric_estimator import BiometricEstimator
from recommendation.workout_generator import WorkoutGenerator
import os

# Create a FastAPI app
app = FastAPI()

# --- Load Models and Helper classes ---
# Ensure you have the model files, scaler, encoder, and tokenizer in the fast_api_app directory
try:
    biometric_estimator = BiometricEstimator(model_path='best_bodym_model.pth')
    
    # Correctly initialize WorkoutGenerator with all required paths
    workout_generator = WorkoutGenerator(
        model_path='trainium_sota_transformer_model.pth',
        tokenizer_path='tokenizer.json',
        scaler_path='scaler.pkl',
        encoder_path='encoder.pkl'
    )
    print("Models and helpers loaded successfully.")
except Exception as e:
    print(f"Error loading models or helpers: {e}")
    biometric_estimator = None
    workout_generator = None

class UserData(BaseModel):
    Age: int
    Gender: str
    height_cm: float
    weight_kg: float
    Goal: str
    level: str
    BMI: float
    chest_cm: float
    waist_cm: float
    hip_cm: float
    thigh_cm: float
    bicep_cm: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Fitness Coach API!"}

@app.post("/predict_biometrics")
async def predict_biometrics(frontal_image: UploadFile = File(...), side_image: UploadFile = File(...)):
    if not frontal_image.content_type.startswith('image/') or not side_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="One or both files are not images.")

    frontal_image_data = await frontal_image.read()
    side_image_data = await side_image.read()

    try:
        biometrics = biometric_estimator.predict(frontal_image_data, side_image_data)
        return {"biometrics": biometrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_workout")
async def generate_workout(user_data: UserData):
    if not workout_generator:
        raise HTTPException(status_code=500, detail="Workout generator not available.")
    try:
        workout_plan = workout_generator.generate_workout_plan(user_data.dict())
        return {"workout_plan": workout_plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)