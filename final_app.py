import gradio as gr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import json
import re
import math
import os
import cv2
import rembg
from PIL import Image
import io
from torchvision import models, transforms
from groq import Groq
import tempfile

# --- 1. Global Setup: Load All Models and Artifacts ---
print("--- Initializing Trainium AI Coach ---")
print("This will take a moment as models are loaded into memory...")

# --- A. Configure Device ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Master Exercise Database for Post-Processing Logic ---
EXERCISE_DATABASE = {
    'SQUAT': ['Legs', 'Strength', 1], 'PUSH-UP': ['Push', 'Strength', 1],
    'PLANK': ['Core', 'Strength', 1], 'JUMPING JACKS': ['Full Body', 'Cardio', 1],
    'GLUTE BRIDGE': ['Legs', 'Strength', 1], 'BENT OVER ROW': ['Pull', 'Strength', 1],
    'WALL SIT': ['Legs', 'Strength', 1], 'BIRD-DOG': ['Core', 'Strength', 1],
    'CRUNCHES': ['Core', 'Strength', 1], 'LEG RAISES': ['Core', 'Strength', 1],
    'WALL PUSH-UPS': ['Push', 'Strength', 1], 'ARM CIRCLES': ['Push', 'Flexibility', 1],
    'SIDE BEND': ['Core', 'Flexibility', 1], 'FORWARD FOLD': ['Legs', 'Flexibility', 1],
    'CAT-COW STretch': ['Core', 'Flexibility', 1], "CHILD'S POSE": ['Full Body', 'Flexibility', 1],
    'COBRA POSE': ['Core', 'Flexibility', 1], 'BOXER SHUFFLE': ['Full Body', 'Cardio', 1],
    'HIGH KNEES': ['Full Body', 'Cardio', 1], 'DONKEY KICKS': ['Legs', 'Strength', 1],
    'FIRE HYDRANTS': ['Legs', 'Strength', 1], 'LUNGE': ['Legs', 'Strength', 2],
    'OVERHEAD PRESS': ['Push', 'Strength', 2], 'TRICEP DIPS': ['Push', 'Strength', 2],
    'CALF RAISES': ['Legs', 'Strength', 2], 'BICEP CURL': ['Pull', 'Strength', 2],
    'RUSSIAN TWIST': ['Core', 'Strength', 2], 'SIDE LUNGES': ['Legs', 'Strength', 2],
    'SUPERMAN': ['Core', 'Strength', 2], 'SIDE PLANK': ['Core', 'Strength', 2],
    'LATERAL RAISES': ['Push', 'Strength', 2], 'SUMO SQUAT': ['Legs', 'Strength', 2],
    'REVERSE CRUNCHES': ['Core', 'Strength', 2], 'PLANK JACKS': ['Core', 'Cardio', 2],
    'GOOD MORNINGS': ['Legs', 'Strength', 2], 'SHOULDER TAPS': ['Core', 'Strength', 2],
    'DOWNWARD DOG': ['Full Body', 'Flexibility', 2], 'FLUTTER KICKS': ['Core', 'Strength', 2],
    'SCISSOR KICKS': ['Core', 'Strength', 2], 'INCHWORM': ['Full Body', 'Strength', 2],
    'HIGH PLANK TO LOW PLANK': ['Push', 'Strength', 2], 'MOUNTAIN CLIMBER': ['Full Body', 'Cardio', 2],
    'DEADLIFT': ['Full Body', 'Strength', 3], 'PULL-UPS': ['Pull', 'Strength', 3],
    'BURPEES': ['Full Body', 'Cardio', 3], 'PIKE PUSH-UP': ['Push', 'Strength', 3],
    'DIAMOND PUSH-UP': ['Push', 'Strength', 3], 'T-POSE HOLD': ['Push', 'Strength', 3]
}

# --- B. Load Vision Model and its Artifacts ---
print("Loading Biometric Estimation Model...")
class BodyM_MetricEstimator(nn.Module):
    def __init__(self, num_measurements):
        super(BodyM_MetricEstimator, self).__init__()
        self.frontal_branch = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.side_branch = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        num_features = self.frontal_branch.classifier[1].in_features
        self.frontal_branch.classifier = nn.Identity()
        self.side_branch.classifier = nn.Identity()
        self.regression_head = nn.Sequential(
            nn.Linear(num_features * 2, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_measurements)
        )
    def forward(self, frontal_img, side_img):
        frontal_features = self.frontal_branch(frontal_img)
        side_features = self.side_branch(side_img)
        combined_features = torch.cat((frontal_features, side_features), dim=1)
        return self.regression_head(combined_features)

VISION_MODEL_TARGET_COLUMNS = [
    'height_cm', 'weight_kg', 'ankle', 'arm-length', 'bicep', 'calf', 'chest',
    'forearm', 'hip', 'leg-length', 'shoulder-breadth', 'shoulder-to-crotch',
    'thigh', 'waist', 'wrist'
]
NUM_TARGETS = len(VISION_MODEL_TARGET_COLUMNS)
vision_model = BodyM_MetricEstimator(num_measurements=NUM_TARGETS)
vision_model.load_state_dict(torch.load('fast_api_app/best_bodym_model.pth', map_location=DEVICE, weights_only=True))
vision_model.to(DEVICE)
vision_model.eval()
print("✅ Biometric Model loaded.")

# --- C. Load Workout Generator Model and its Artifacts ---
print("Loading Workout Generation Model...")
class PyTorchTokenizer:
    def __init__(self, oov_token='<unk>'):
        self.word_to_idx = {}
        self.idx_to_word = {}
    @classmethod
    def load_tokenizer(cls, path):
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer.word_to_idx = json.load(f)
        tokenizer.idx_to_word = {i: w for w, i in tokenizer.word_to_idx.items()}
        tokenizer.vocab_size = len(tokenizer.word_to_idx)
        return tokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class WorkoutGenerationTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_decoder_layers, dim_feedforward, vocab_size, max_exercises):
        super(WorkoutGenerationTransformer, self).__init__()
        self.d_model, self.max_exercises, self.vocab_size = d_model, max_exercises, vocab_size
        self.profile_embedding = nn.Linear(input_dim, d_model)
        self.exercise_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, max_exercises * vocab_size)
    def forward(self, src_profile, trg_sequence):
        memory = self.profile_embedding(src_profile).unsqueeze(1).repeat(1, 30, 1)
        trg_flat = trg_sequence.view(-1, self.max_exercises)
        embedded_trg = self.exercise_embedding(trg_flat).mean(dim=1).view(src_profile.shape[0], 30, self.d_model)
        embedded_trg = self.pos_encoder(embedded_trg)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(trg_sequence.size(1)).to(DEVICE)
        transformer_output = self.transformer_decoder(tgt=embedded_trg, memory=memory, tgt_mask=tgt_mask)
        output = self.fc_out(transformer_output).view(src_profile.shape[0], 30, self.max_exercises, self.vocab_size)
        return output

workout_scaler = joblib.load('fast_api_app/scaler.pkl')
workout_encoder = joblib.load('fast_api_app/encoder.pkl')
workout_tokenizer = PyTorchTokenizer.load_tokenizer('fast_api_app/tokenizer.json')
VOCAB_SIZE = workout_tokenizer.vocab_size
INPUT_DIM, D_MODEL, NHEAD, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_EXERCISES_PER_DAY = 18, 512, 8, 6, 2048, 20
workout_model = WorkoutGenerationTransformer(
    INPUT_DIM, D_MODEL, NHEAD, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, VOCAB_SIZE, MAX_EXERCISES_PER_DAY
).to(DEVICE)
workout_model.load_state_dict(torch.load('fast_api_app/trainium_sota_transformer_model.pth', map_location=DEVICE, weights_only=True))
workout_model.eval()
print("✅ Workout Generation Model loaded.")
print("\n--- Trainium is ready ---")

# --- 2. Define Helper and Prediction Functions ---

def _process_image_to_silhouette(image_array):
    try:
        foreground = rembg.remove(Image.fromarray(image_array))
        foreground_cv = np.array(foreground)
        gray = cv2.cvtColor(foreground_cv, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        silhouette = cv2.bitwise_not(mask)
        return cv2.resize(silhouette, (512, 512))
    except Exception as e:
        print(f"Error processing image for silhouette: {e}")
        return None

def predict_metrics(frontal_img, side_img):
    frontal_silhouette = _process_image_to_silhouette(frontal_img)
    side_silhouette = _process_image_to_silhouette(side_img)
    if frontal_silhouette is None or side_silhouette is None:
        raise ValueError("Could not process one or both images into silhouettes.")
    image_transforms = transforms.Compose([
        transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    frontal_tensor = image_transforms(frontal_silhouette).unsqueeze(0).to(DEVICE)
    side_tensor = image_transforms(side_silhouette).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predictions = vision_model(frontal_tensor, side_tensor)
    return dict(zip(VISION_MODEL_TARGET_COLUMNS, predictions.squeeze(0).cpu().numpy()))

def top_p_sampling(logits, p=0.92, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    probs[indices_to_remove] = 0.0
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)

def generate_plan_from_metrics(metrics, goal, level, age, gender):
    bmi = round(metrics['weight_kg'] / ((metrics['height_cm'] / 100) ** 2), 1)
    user_profile = {
        'Age': age, 'Gender': gender, 'Goal': goal, 'level': level,
        'height_cm': metrics['height_cm'], 'weight_kg': metrics['weight_kg'], 'BMI': bmi,
        'chest_cm': metrics.get('chest', 100), 'waist_cm': metrics.get('waist', 85),
        'hip_cm': metrics.get('hip', 95), 'thigh_cm': metrics.get('thigh', 55),
        'bicep_cm': metrics.get('bicep', 35)
    }
    NUMERICAL_COLS = ['Age', 'height_cm', 'weight_kg', 'BMI', 'chest_cm', 'waist_cm', 'hip_cm', 'thigh_cm', 'bicep_cm']
    CATEGORICAL_COLS = ['Gender', 'Goal', 'level']
    user_df = pd.DataFrame([user_profile])
    user_cat = workout_encoder.transform(user_df[CATEGORICAL_COLS])
    user_num = workout_scaler.transform(user_df[NUMERICAL_COLS])
    user_processed = np.concatenate([user_num, user_cat], axis=1)
    src_tensor = torch.tensor(user_processed, dtype=torch.float32).to(DEVICE)
    
    generated_sequence = torch.zeros((1, 30, MAX_EXERCISES_PER_DAY), dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        for i in range(30):
            output_logits = workout_model(src_tensor, generated_sequence)
            day_logits = output_logits[:, i, :, :]
            day_predictions = [top_p_sampling(day_logits[:, j, :]).item() for j in range(MAX_EXERCISES_PER_DAY)]
            generated_sequence[:, i, :] = torch.tensor(day_predictions, device=DEVICE).long()
    
    predicted_plan_ids = generated_sequence.cpu().numpy()
    
    plan_dict = {"Day": [], "Workout": []}
    rest_day_string = "Rest Day"
    for i in range(30):
        day_number = i + 1
        plan_dict["Day"].append(f"Day {day_number}")
        if day_number % 7 == 0:
            plan_dict["Workout"].append(rest_day_string)
        else:
            day_tokens = predicted_plan_ids[0, i, :]
            raw_exercises = [workout_tokenizer.idx_to_word.get(token, '') for token in day_tokens if token != 0]
            compulsory_warmup = ['JUMPING JACKS', 'ARM CIRCLES']
            compulsory_cooldown = ['PLANK', 'COBRA POSE']
            tokens_to_remove = {rest_day_string, '<unk>', ''}
            compulsory_set = set(compulsory_warmup + compulsory_cooldown)
            model_suggestions = [ex for ex in raw_exercises if ex not in tokens_to_remove and ex not in compulsory_set]
            unique_suggestions = list(dict.fromkeys(model_suggestions))
            main_workout_suggestions = []
            other_stretches = []
            for ex in unique_suggestions:
                if EXERCISE_DATABASE.get(ex, [None, 'Strength'])[1] in ['Strength', 'Cardio']:
                    main_workout_suggestions.append(ex)
                else:
                    other_stretches.append(ex)
            main_workout_suggestions.sort(key=lambda ex: EXERCISE_DATABASE.get(ex, [None, None, 99])[2])
            final_plan_list = compulsory_warmup + main_workout_suggestions + other_stretches + compulsory_cooldown
            final_workout = ", ".join(final_plan_list)
            plan_dict["Workout"].append(final_workout)
            
    return pd.DataFrame(plan_dict), user_profile

# --- 3. Main Gradio Function (Orchestrator) ---
def full_pipeline(frontal_img, side_img, goal, level, age, gender, progress=gr.Progress()):
    if frontal_img is None or side_img is None:
        raise gr.Error("Please upload both a frontal and a side view image.")
    try:
        progress(0, desc="[Step 1/2] Analyzing images to estimate body metrics...")
        predicted_metrics = predict_metrics(frontal_img, side_img)
        progress(0.5, desc="[Step 2/2] Generating your personalized 30-day workout plan...")
        final_plan, user_profile = generate_plan_from_metrics(predicted_metrics, goal, level, age, gender)
        
        temp_dir = tempfile.mkdtemp()
        csv_file_path = os.path.join(temp_dir, "my_30day_workout_plan.csv")
        final_plan.to_csv(csv_file_path, index=False)
        
        return final_plan, user_profile, final_plan, gr.update(value=csv_file_path, visible=True)
    
    except ValueError as e:
        raise gr.Error(f"Image Processing Error: {e}")
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {e}")

# --- NEW: Function to get AI Narrative from Groq ---
def get_ai_narrative(user_profile, workout_plan_df, progress=gr.Progress()):
    progress(0, desc="Asking Trainium AI for insights...")
    if user_profile is None or workout_plan_df is None:
        return "Please generate a workout plan first before asking for insights."

    # Format the plan and profile for the LLM
    plan_str = workout_plan_df.to_string(index=False)
    profile_str = "\n".join([f"- {key}: {value}" for key, value in user_profile.items()])

    system_prompt = (
        "You are 'Trainium', an expert AI fitness coach. You are friendly, motivating, and knowledgeable. "
        "Your task is to provide a comprehensive, easy-to-follow, and inspiring guide based on the user's profile and generated workout plan. "
        "Follow the user's requested structure precisely. Use markdown for clear formatting (headings, bold text, lists)."
    )
    
    ### MODIFIED ###: Changed the prompt to remove the hardcoded example and give better instructions.
    user_prompt = (
        f"Here is my profile:\\n{profile_str}\\n\\n"
        f"And here is the 30-day workout plan you generated for me:\\n{plan_str}\\n\\n"
        "Please generate a detailed response following this exact structure in order:\n"
        "i) A welcome message to Trainium and a brief, personalized introduction to how the AI model works.\n"
        "ii) An explanation of my current body state and fitness level based on my profile. IMPORTANT: Do not mention my raw weight, height and BMI anywhere in the output. Instead, list my other body measurements (chest, waist, hip, thigh, bicep) from the profile provided. For each measurement, state its value and compare it to general averages for my stated gender (e.g., 'average', 'above average'). Format this as a list, for example: '+ Chest: [value from profile] cm ([comparison])'.\n"
        "iii) Your suggestion for my ideal goal metrics. This section MUST include specific numerical targets for 'Target Weight', 'Target BMI', and 'Target Body Fat Percentage'.(do not compare with the original weights)\n"
        "iv) Detailed pointwise advice on how to achieve this goal using the provided plan.\n"
        "v) Suggestions for lifestyle changes (like sleep, hydration, etc.).\n"
        "vi) A sample 7-day diet plan (including snacks and meals) for me to follow.\n"
        "vii) A final, powerful motivational message to inspire me to become a better version of myself."
    )

    try:
        progress(0.5, desc="Generating detailed guide...")

        # Load API key from environment variable instead of hardcoding it
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise gr.Error("GROQ_API_KEY environment variable not set.")
        client = Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        error_message = f"An error occurred while communicating with the AI. Please check your API key and network connection. Details: {e}"
        print(error_message)
        raise gr.Error(error_message)


# --- 4. Create and Launch the Gradio Interface ---
custom_css = ".fit-content-button { width: fit-content !important; }"

with gr.Blocks(theme=gr.themes.Soft(), title="Trainium AI Coach", css=custom_css) as demo:
    user_profile_state = gr.State()
    workout_plan_state = gr.State()

    gr.Markdown("# 🏋️ Trainium: AI Workout Plan Generator (Full Pipeline)")
    gr.Markdown("Upload your images, select your goals, and let the AI build your plan.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Your Images")
            frontal_image = gr.Image(type="numpy", label="Frontal View Photo", height=250)
            side_image = gr.Image(type="numpy", label="Side View Photo", height=250)
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Define Your Profile")
            age = gr.Slider(label="Your Age", minimum=16, maximum=80, value=30, step=1)
            gender = gr.Dropdown(label="Gender", choices=["Male", "Female"], value="Male")
            goal = gr.Dropdown(label="Primary Goal", choices=["Lose Weight", "Gain Muscle", "Gain Stamina", "General Fitness"], value="General Fitness")
            level = gr.Dropdown(label="Your Fitness Level", choices=["Beginner", "Intermediate", "Advanced"], value="Intermediate")
            generate_btn = gr.Button("Generate My Plan", variant="primary", size="lg")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Your 30-Day Plan")
            
            gr.Markdown("##### download workout plan as a csv")
            
            export_btn = gr.DownloadButton(
                "export workout plan", 
                visible=False, 
                variant="secondary",
                elem_classes=["fit-content-button"]
            )
            
            output_plan = gr.Dataframe(
                headers=["Day", "Workout"], 
                datatype=["str", "str"], 
                row_count=30, 
                col_count=(2, "fixed"),
                column_widths=["15%", "85%"],
                wrap=True
            )
            
            ask_ai_btn = gr.Button("Ask Trainium AI for Insights", variant="secondary")
            ai_narrative_output = gr.Markdown(visible=False)

    generate_btn.click(
        fn=full_pipeline,
        inputs=[frontal_image, side_image, goal, level, age, gender],
        outputs=[output_plan, user_profile_state, workout_plan_state, export_btn]
    )

    ask_ai_btn.click(
        fn=lambda: gr.update(visible=True),
        outputs=ai_narrative_output
    ).then(
        fn=get_ai_narrative,
        inputs=[user_profile_state, workout_plan_state],
        outputs=ai_narrative_output
    )


if __name__ == "__main__":
    demo.launch()