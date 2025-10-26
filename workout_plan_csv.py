import csv
import random
import pandas as pd
import numpy as np

# --- Configuration ---
NUM_ROWS_TO_GENERATE = 100000
OUTPUT_FILE_NAME = "trainium_production_dataset_100k.csv"

# --- Master Exercise Database with Difficulty Tiers ---
# [Primary Muscle Group, Type, Difficulty (1: Beginner, 2: Intermediate, 3: Advanced)]
EXERCISE_DATABASE = {
    # --- Beginner Tier (1) ---
    'SQUAT': ['Legs', 'Strength', 1],
    'PUSH-UP': ['Push', 'Strength', 1],
    'PLANK': ['Core', 'Strength', 1],
    'JUMPING JACKS': ['Full Body', 'Cardio', 1],
    'GLUTE BRIDGE': ['Legs', 'Strength', 1],
    'BENT OVER ROW': ['Pull', 'Strength', 1],
    'WALL SIT': ['Legs', 'Strength', 1],
    'BIRD-DOG': ['Core', 'Strength', 1],
    'CRUNCHES': ['Core', 'Strength', 1],
    'LEG RAISES': ['Core', 'Strength', 1],
    'WALL PUSH-UPS': ['Push', 'Strength', 1],
    'ARM CIRCLES': ['Push', 'Flexibility', 1],
    'SIDE BEND': ['Core', 'Flexibility', 1],
    'FORWARD FOLD': ['Legs', 'Flexibility', 1],
    'CAT-COW STRETCH': ['Core', 'Flexibility', 1],
    "CHILD'S POSE": ['Full Body', 'Flexibility', 1],
    'COBRA POSE': ['Core', 'Flexibility', 1],
    'BOXER SHUFFLE': ['Full Body', 'Cardio', 1],
    'HIGH KNEES': ['Full Body', 'Cardio', 1],
    'DONKEY KICKS': ['Legs', 'Strength', 1],
    'FIRE HYDRANTS': ['Legs', 'Strength', 1],

    # --- Intermediate Tier (2) ---
    'LUNGE': ['Legs', 'Strength', 2],
    'OVERHEAD PRESS': ['Push', 'Strength', 2],
    'TRICEP DIPS': ['Push', 'Strength', 2],
    'CALF RAISES': ['Legs', 'Strength', 2],
    'BICEP CURL': ['Pull', 'Strength', 2],
    'RUSSIAN TWIST': ['Core', 'Strength', 2],
    'SIDE LUNGES': ['Legs', 'Strength', 2],
    'SUPERMAN': ['Core', 'Strength', 2],
    'SIDE PLANK': ['Core', 'Strength', 2],
    'LATERAL RAISES': ['Push', 'Strength', 2],
    'SUMO SQUAT': ['Legs', 'Strength', 2],
    'REVERSE CRUNCHES': ['Core', 'Strength', 2],
    'PLANK JACKS': ['Core', 'Cardio', 2],
    'GOOD MORNINGS': ['Legs', 'Strength', 2],
    'SHOULDER TAPS': ['Core', 'Strength', 2],
    'DOWNWARD DOG': ['Full Body', 'Flexibility', 2],
    'FLUTTER KICKS': ['Core', 'Strength', 2],
    'SCISSOR KICKS': ['Core', 'Strength', 2],
    'INCHWORM': ['Full Body', 'Strength', 2],
    'HIGH PLANK TO LOW PLANK': ['Push', 'Strength', 2],
    'MOUNTAIN CLIMBER': ['Full Body', 'Cardio', 2],

    # --- Advanced Tier (3) ---
    'DEADLIFT': ['Full Body', 'Strength', 3],
    'PULL-UPS': ['Pull', 'Strength', 3],
    'BURPEES': ['Full Body', 'Cardio', 3],
    'PIKE PUSH-UP': ['Push', 'Strength', 3],
    'DIAMOND PUSH-UP': ['Push', 'Strength', 3],
    'T-POSE HOLD': ['Push', 'Strength', 3]
}

# --- Helper Functions ---
def get_exercises(difficulty_tiers, muscle_group=None, ex_type=None, count=1):
    """Fetches a specified number of unique exercises matching the criteria."""
    pool = [
        name for name, props in EXERCISE_DATABASE.items()
        if props[2] in difficulty_tiers and
           (muscle_group is None or props[0] in muscle_group) and
           (ex_type is None or props[1] in ex_type)
    ]
    if not pool: return []
    return random.sample(pool, min(len(pool), count))

# --- The Core Workout Generation Engine ---
def generate_workout_plan(goal, user_level):
    """Generates a structured, progressive 30-day workout plan based on user's stated level."""
    plan = [""] * 30
    schedules = {
        'Gain Muscle': ['Push', 'Pull', 'Legs', 'Rest', 'Push', 'Pull', 'Rest'],
        'Lose Weight': ['Full Body Strength', 'HIIT', 'Full Body Strength', 'Active Recovery', 'Core & Cardio', 'HIIT', 'Rest'],
        'Gain Stamina': ['Cardio', 'HIIT', 'Cardio', 'Active Recovery', 'Cardio Intervals', 'Long Cardio', 'Rest'],
        'General Fitness': ['Upper Body', 'Lower Body', 'Full Body', 'Rest', 'Upper Body', 'Lower Body', 'Rest']
    }
    schedule = schedules[goal]

    # --- NEW: Level-dependent volume scaling ---
    if user_level == 'Beginner':
        start_exercise_count, end_exercise_count = 8, 15
    elif user_level == 'Intermediate':
        start_exercise_count, end_exercise_count = 10, 18
    else: # Advanced
        start_exercise_count, end_exercise_count = 12, 20

    for day_index in range(30):
        day_type = schedule[day_index % 7]
        week = day_index // 7
        
        # Dynamic Volume Progression Logic
        progress_ratio = day_index / 29.0
        target_total_exercises = int(start_exercise_count + (end_exercise_count - start_exercise_count) * progress_ratio)

        is_deload_week = (week == 3)
        if is_deload_week:
            # During deload week, volume drops to ~60-70% of the previous week's target
            target_total_exercises = int(target_total_exercises * 0.65)

        # Determine allowed exercise difficulty based on user's selected level and week
        if user_level == 'Beginner':
            allowed_tiers = [1] if week < 2 else [1, 2]
        elif user_level == 'Intermediate':
            allowed_tiers = [1, 2] if week < 2 else [1, 2, 3]
        else: # Advanced
            allowed_tiers = [1, 2] if week == 0 else [1, 2, 3]

        # Assemble the workout for the day
        workout = []
        if day_type == 'Rest':
            plan[day_index] = 'Rest Day'
            continue
        if day_type == 'Active Recovery':
            plan[day_index] = ", ".join(get_exercises(difficulty_tiers=[1], ex_type=['Flexibility'], count=5))
            continue

        warmup_count = 2
        workout += get_exercises(difficulty_tiers=[1], ex_type=['Cardio', 'Flexibility'], count=warmup_count)
        
        accessory_count = int(2 + (2 * progress_ratio))
        if is_deload_week: accessory_count = 2
        
        main_count = target_total_exercises - warmup_count - accessory_count
        if main_count < 1: main_count = 1
        
        if day_type in ['Push', 'Upper Body']:
            workout += get_exercises(allowed_tiers, muscle_group=['Push'], count=main_count)
        elif day_type == 'Pull':
             workout += get_exercises(allowed_tiers, muscle_group=['Pull'], count=main_count)
        elif day_type in ['Legs', 'Lower Body']:
            workout += get_exercises(allowed_tiers, muscle_group=['Legs'], count=main_count)
        elif day_type in ['Full Body Strength', 'Full Body']:
             main_pool = get_exercises(allowed_tiers, muscle_group=['Push', 'Pull', 'Legs', 'Full Body'])
             workout += random.sample(main_pool, min(len(main_pool), main_count))
        elif day_type in ['HIIT', 'Cardio Intervals']:
            workout += get_exercises(allowed_tiers, ex_type=['Cardio'], count=main_count)
        elif day_type in ['Core & Cardio', 'Cardio', 'Long Cardio']:
            cardio_main_count = (main_count // 2) + (main_count % 2)
            core_main_count = main_count // 2
            workout += get_exercises(allowed_tiers, ex_type=['Cardio'], count=cardio_main_count)
            workout += get_exercises(allowed_tiers, muscle_group=['Core'], count=core_main_count)

        workout += get_exercises(allowed_tiers, muscle_group=['Core'], count=accessory_count)

        final_workout = sorted(list(set(workout)))
        plan[day_index] = ", ".join(final_workout)
        
    return plan

# --- Data Profile Generation ---
def calculate_bmi(height_m, weight_kg):
    return round(weight_kg / (height_m ** 2), 1)

def generate_data(num_rows):
    """Generates the specified number of user profiles and their workout plans."""
    data = []
    for user_id in range(1, num_rows + 1):
        age = random.randint(18, 65)
        gender = random.choice(['Male', 'Female'])
        goal = random.choice(['Lose Weight', 'Gain Muscle', 'Gain Stamina', 'General Fitness'])

        if gender == 'Male':
            height_cm = random.uniform(165, 195)
            weight_kg = random.uniform(60, 130)
        else:
            height_cm = random.uniform(150, 180)
            weight_kg = random.uniform(45, 110)
            
        bmi = calculate_bmi(height_cm / 100, weight_kg)

        # --- NEW: Smarter level selection based on profile ---
        if bmi > 29 or age > 55:
            level = random.choices(['Beginner', 'Intermediate', 'Advanced'], weights=[0.7, 0.25, 0.05], k=1)[0]
        elif bmi > 25 or age > 40:
            level = random.choices(['Beginner', 'Intermediate', 'Advanced'], weights=[0.4, 0.5, 0.1], k=1)[0]
        else:
            level = random.choices(['Beginner', 'Intermediate', 'Advanced'], weights=[0.1, 0.5, 0.4], k=1)[0]

        row = {
            'UserID': user_id,
            'Age': age,
            'Gender': gender,
            'Goal': goal,
            'level': level,
            'height_cm': round(height_cm, 2),
            'weight_kg': round(weight_kg, 2),
            'BMI': bmi,
        }
        
        workout_plan = generate_workout_plan(goal, level)
        for i, workout in enumerate(workout_plan):
            row[f'Day_{i+1}'] = workout
            
        data.append(row)
        if user_id % 1000 == 0:
            print(f"Generated {user_id}/{num_rows} rows...")
            
    return data

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"🚀 Starting FINAL production data generation for {NUM_ROWS_TO_GENERATE} users...")
    
    dataset = generate_data(NUM_ROWS_TO_GENERATE)
    df = pd.DataFrame(dataset)
    
    # Use NumPy (np) to generate correlated body metrics
    df['chest_cm'] = (df['height_cm'] * 0.55 + (df['weight_kg'] - 70) * 0.5 + np.random.uniform(-5, 5, df.shape[0])).round(2)
    df['waist_cm'] = (df['height_cm'] * 0.48 + (df['weight_kg'] - 70) * 0.8 + np.random.uniform(-5, 5, df.shape[0])).round(2)
    df['hip_cm'] = (df['height_cm'] * 0.58 + (df['weight_kg'] - 70) * 0.4 + np.random.uniform(-5, 5, df.shape[0])).round(2)
    df['thigh_cm'] = (df['height_cm'] * 0.34 + (df['weight_kg'] - 70) * 0.3 + np.random.uniform(-5, 5, df.shape[0])).round(2)
    df['bicep_cm'] = (df['height_cm'] * 0.20 + (df['weight_kg'] - 70) * 0.2 + np.random.uniform(-3, 3, df.shape[0])).round(2)
    
    # Reorder columns for clarity
    cols = ['UserID', 'Age', 'Gender', 'Goal', 'level', 'height_cm', 'weight_kg', 'BMI', 'chest_cm', 'waist_cm', 'hip_cm', 'thigh_cm', 'bicep_cm']
    day_cols = [f'Day_{i}' for i in range(1, 31)]
    df = df[cols + day_cols]
    
    df.to_csv(OUTPUT_FILE_NAME, index=False)
    
    print(f"\n✅ Success! {NUM_ROWS_TO_GENERATE} rows of production-grade data generated.")
    print(f"File saved as: {OUTPUT_FILE_NAME}")

