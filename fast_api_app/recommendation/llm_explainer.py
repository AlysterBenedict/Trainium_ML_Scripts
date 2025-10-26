# llm_explainer.py

import os
from groq import Groq

def get_workout_narrative(structured_plan, user_profile):
    """
    Uses Groq's LLM to generate a detailed, friendly explanation of the workout plan.
    """
    try:
        # Load API key from environment variable instead of hardcoding it
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")

        client = Groq(api_key=api_key)
    except Exception as e:
        return f"Error initializing Groq client: {e}"

    # Convert the plan dictionary to a more readable string format
    plan_str = ""
    for day, exercises in structured_plan.items():
        if exercises: # Only include days with exercises
            plan_str += f"{day.replace('_', ' ')}: {', '.join(exercises)}\n"

    system_prompt = (
        "You are 'Trainium', an expert AI fitness coach. You are friendly, motivating, and knowledgeable. "
        "Your task is to take a structured 30-day workout plan and a user's profile and transform it into a "
        "comprehensive, easy-to-follow, and inspiring guide. "
        "Explain the 'why' behind the plan's structure. For example, explain the split (e.g., Push/Pull/Legs), "
        "the importance of rest days, and how the plan progresses. "
        "Provide tips on form for a few key exercises. Add a concluding motivational message. "
        "Use markdown for clear formatting (headings, bold text, lists)."
    )

    user_prompt = (
        f"Here is my profile:\n"
        f"- Goal: {user_profile['Goal']}\n"
        f"- Experience Level: {user_profile['level']}\n\n"
        f"And here is the 30-day workout plan you generated for me:\n"
        f"{plan_str}\n\n"
        f"Please provide a detailed and motivating explanation of this plan."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile", # Using a powerful model for high-quality text
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while communicating with the Groq API: {e}"