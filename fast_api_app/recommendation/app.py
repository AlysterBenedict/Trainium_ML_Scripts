# main.py

import os
import torch
import warnings
from biometric_estimator import predict_metrics_from_images
from workout_generator import generate_workout_plan
from llm_explainer import get_workout_narrative

# Suppress known PyTorch warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    """Main pipeline to run the full AI workout generation process."""

    # --- 1. Define User Inputs & Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARTIFACTS_PATH = os.path.join(os.getcwd(), "models")

    # ✅ Use raw strings for Windows paths (no f-strings!)
    VISION_MODEL_PATH = r"C:\Users\bened\Documents\Alyster Coding\PROJECTS\AI_GYM\best_bodym_model.pth"
    user_frontal_photo = r"C:\Users\bened\Documents\Alyster Coding\PROJECTS\AI_GYM\imgs\alvin2.jpg"
    user_side_photo = r"C:\Users\bened\Documents\Alyster Coding\PROJECTS\AI_GYM\imgs\alvin1.jpg"

    # --- File Existence Validation ---
    for path in [VISION_MODEL_PATH, user_frontal_photo, user_side_photo]:
        if not os.path.exists(path):
            print(f"❌ Error: Required file not found -> {path}")
            return

    # --- User Info (can be expanded later) ---
    user_provided_info = {
        "Age": 30,
        "Gender": "Male",          # 'Male' or 'Female'
        "Goal": "Muscle Gain",     # 'Muscle Gain', 'Fat Loss', 'General Fitness'
        "level": "Intermediate"    # 'Beginner', 'Intermediate', 'Advanced'
    }

    print("=" * 60)
    print("🤖 Welcome to the AI Personalized Workout Generator 🤖")
    print("=" * 60)
    print(f"Using device: {DEVICE}")

    # --- 2. Step 1: Predict Body Metrics from Images ---
    print("\n--- [Step 1/4] Analyzing your photos to estimate body metrics... ---")
    try:
        predicted_metrics = predict_metrics_from_images(
            frontal_path=user_frontal_photo,
            side_path=user_side_photo,
            model_path=VISION_MODEL_PATH,
            device=DEVICE
        )
        print("✅ Biometric analysis complete.")
        for key, val in predicted_metrics.items():
            print(f"  - {key}: {val:.2f}")
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error in Step 1: {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error during biometric estimation: {e}")
        return

    # --- 3. Step 2: Combine All User Data ---
    print("\n--- [Step 2/4] Compiling your complete user profile... ---")
    try:
        full_user_profile = user_provided_info.copy()
        full_user_profile.update({
            "height_cm": predicted_metrics["height_cm"],
            "weight_kg": predicted_metrics["weight_kg"],
            "chest_cm": predicted_metrics["chest"],
            "waist_cm": predicted_metrics["waist"],
            "hip_cm": predicted_metrics["hip"],
            "thigh_cm": predicted_metrics["thigh"],
            "bicep_cm": predicted_metrics["bicep"],
        })

        # Compute BMI safely
        height_m = full_user_profile["height_cm"] / 100
        weight_kg = full_user_profile["weight_kg"]
        full_user_profile["BMI"] = round(weight_kg / (height_m ** 2), 2)

        print("✅ Profile compiled successfully.")
        for k, v in full_user_profile.items():
            print(f"  - {k}: {v}")
    except KeyError as e:
        print(f"❌ Missing metric in predicted data: {e}")
        return
    except Exception as e:
        print(f"❌ Error in Step 2: {e}")
        return

    # --- 4. Step 3: Generate the Structured Workout Plan ---
    print("\n--- [Step 3/4] Generating your personalized 30-day workout plan... ---")
    try:
        structured_plan = generate_workout_plan(
            user_profile=full_user_profile,
            artifacts_path=ARTIFACTS_PATH,
            device=DEVICE
        )
        print("✅ Structured workout plan generated.")
    except Exception as e:
        print(f"❌ Error in Step 3: Could not generate workout plan. {e}")
        return

    # --- 5. Step 4: Get Detailed Explanation from LLM ---
    print("\n--- [Step 4/4] Creating your detailed workout guide with tips... ---")
    try:
        detailed_narrative = get_workout_narrative(structured_plan, full_user_profile)
        print("✅ Detailed guide created.")
    except Exception as e:
        print(f"❌ Error in Step 4: Could not get explanation from LLM. {e}")
        # Fallback: show structured plan
        detailed_narrative = "⚠️ Could not generate a detailed narrative. Here's your structured plan:\n"
        for day, exs in structured_plan.items():
            detailed_narrative += f"{day}: {', '.join(exs)}\n"

    # --- 6. Final Output ---
    print("\n" + "=" * 60)
    print("🎉 Your Personalized AI Workout Plan is Ready! 🎉")
    print("=" * 60)
    print(detailed_narrative)
    print("=" * 60)


if __name__ == "__main__":
    main()
