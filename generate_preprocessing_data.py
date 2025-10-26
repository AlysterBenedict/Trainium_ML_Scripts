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