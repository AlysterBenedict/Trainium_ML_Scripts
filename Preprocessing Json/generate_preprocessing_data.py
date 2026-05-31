# NOTE: While this file is not imported by other Python scripts in this backend (which load the
# binary 'scaler.pkl' and 'encoder.pkl' using joblib), it is highly necessary for the overall codebase.
#
# WHY IT IS NECESSARY & WHERE IT IS USED:
# 1. Frontend & Client Compatibility: It exports MinMaxScaler ('min_', 'scale_') and LabelEncoder ('classes_')
#    data to a standard, cross-platform 'preprocessing_data.json' format. This is crucial for clients (like
#    the Kotlin-based Android app 'Trainium.apk') to perform identical feature scaling and label encoding on
#    the client side before sending requests to the API, without needing a Python runtime or scikit-learn.
# 2. Dependency Independence & Debugging: Pickle/joblib files can easily break across different Python/scikit-learn
#    versions. This file ensures the exact preprocessing parameters are human-readable, auditable, and
#    reconstructible in any environment.

import json
import numpy as np

print("--- Manually Generating Preprocessing Data ---")

try:
    # These values are extracted directly from the training source code,
    # bypassing the need to load the incompatible .pkl files.
    
    # 1. Scaler Data (from MinMaxScaler)
    # These are the 'min_' and 'scale_' attributes of the scaler object.
    scaler_data = {
        "min_": [
            18.0, 45.0, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
        "scale_": [
            0.02173913, 0.00952381, 0.01818182, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
    }

    # 2. Encoder Data (from LabelEncoder)
    # This is the 'classes_' attribute of the encoder object.
    encoder_data = {
        "classes_": [
            "Endomorph", "Ectomorph", "Mesomorph"
        ]
    }

    # 3. Combine into a single dictionary
    preprocessing_data = {
        "scaler": scaler_data,
        "encoder": encoder_data
    }

    # 4. Save to a new JSON file
    with open('preprocessing_data.json', 'w') as f:
        json.dump(preprocessing_data, f, indent=4)
        
    print("✅ Success! Data generated and saved to preprocessing_data.json")

except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")

print("\n--- Generation Complete ---")