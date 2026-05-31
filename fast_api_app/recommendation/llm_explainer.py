# =====================================================================================
# LLM Workout Explanation & Narration Module
# =====================================================================================
# USE CASE:
# Connects with a local LM Studio server hosted at 'http://192.168.1.6:1234/v1/chat/completions'
# using model 'google/gemma-4-26b-a4b' to generate a detailed, structured, friendly,
# and highly motivating explanation of the generated 30-day workout plan. It details:
# 1. Custom introduction to how the AI model works.
# 2. Detailed biometric assessments and averages comparison.
# 3. Targets for Weight, BMI, and Body Fat Percentage.
# 4. Pointwise training execution advice, lifestyle suggestions, and a 7-day diet menu.
# 5. Concluding motivational messaging.
#
# CALLED / IMPORTED BY:
# - Pipeline Testing/pipeline_testing_app.py (Multimodal evaluation application)
# =====================================================================================

import requests
import json

def get_workout_narrative(structured_plan, user_profile):
    """
    Connects with the local LM Studio server using 'google/gemma-4-26b-a4b'
    to generate an in-depth, formatted guide based on the user's plan.
    """
    # Convert the plan dictionary to a more readable string format
    plan_str = ""
    for day, exercises in structured_plan.items():
        if exercises: # Only include days with exercises
            plan_str += f"{day.replace('_', ' ')}: {', '.join(exercises)}\n"

    profile_str = "\n".join([f"- {key}: {value}" for key, value in user_profile.items()])

    system_prompt = (
        "You are 'Trainium', an expert AI fitness coach. You are friendly, motivating, and knowledgeable. "
        "Your task is to provide a comprehensive, easy-to-follow, and inspiring guide based on the user's profile and generated workout plan. "
        "Follow the user's requested structure precisely. Use markdown for clear formatting (headings, bold text, lists).\n"
        "CRITICAL: Keep your internal thinking/reasoning process extremely brief (under 50 tokens) and get straight to the final response content."
    )

    user_prompt = (
        f"Here is my profile:\n{profile_str}\n\n"
        f"And here is the 30-day workout plan you generated for me:\n{plan_str}\n\n"
        "Please generate a detailed response following this exact structure in order:\n"
        "i) A welcome message to Trainium and a brief, personalized introduction to how the AI model works.\n"
        "ii) An explanation of my current body state and fitness level based on my profile. IMPORTANT: Mention my raw weight, height and BMI in the output. Also, list my other body measurements (chest, waist, hip, thigh, bicep) from the profile provided. For each measurement, state its value and compare it to general averages for my stated gender (e.g., 'average', 'above average'). Format this as a list, for example: '+ Chest: [value from profile] cm ([comparison])'.\n"
        "iii) Your suggestion for my ideal goal metrics. This section MUST include specific numerical targets for 'Target Weight', 'Target BMI', and 'Target Body Fat Percentage'.(do not compare with the original weights)\n"
        "iv) Detailed pointwise advice on how to achieve this goal using the provided plan.\n"
        "v) Suggestions for lifestyle changes (like sleep, hydration, etc.).\n"
        "vi) A sample 7-day diet plan (including snacks and meals) for me to follow.\n"
        "vii) A final, powerful motivational message to inspire me to become a better version of myself."
    )

    try:
        url = "http://192.168.1.6:1234/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": "google/gemma-4-26b-a4b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 8192
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result_json = response.json()
        return result_json['choices'][0]['message']['content']
    except Exception as e:
        error_message = f"An error occurred while communicating with the local AI. Please check LM Studio connection at http://192.168.1.6:1234 and make sure model 'google/gemma-4-26b-a4b' is loaded. Details: {e}"
        print(error_message)
        return f"### ⚠️ Connection Error\n\n{error_message}"