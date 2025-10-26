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

# --- 1. Load Production Artifacts Globally ---
# This ensures that the model and preprocessors are loaded only ONCE when the app starts,
# making inference much faster.

print("--- Loading Production Artifacts ---")

# --- GPU Verification (Define device globally) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Custom Tokenizer Class (Must be identical to the one used in training) ---
class PyTorchTokenizer:
    def __init__(self, oov_token='<unk>'):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.oov_token = oov_token
        self.word_to_idx['<pad>'] = 0
        self.idx_to_word[0] = '<pad>'
        self.word_to_idx[oov_token] = 1
        self.idx_to_word[1] = oov_token
        self.vocab_size = 2

    @classmethod
    def load_tokenizer(cls, path):
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer.word_to_idx = json.load(f)
        tokenizer.idx_to_word = {i: w for w, i in tokenizer.word_to_idx.items()}
        tokenizer.vocab_size = len(tokenizer.word_to_idx)
        return tokenizer

# --- SOTA Model Architecture (Must be identical to the one used in training) ---
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class WorkoutGenerationTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_decoder_layers, dim_feedforward, vocab_size, max_exercises):
        super(WorkoutGenerationTransformer, self).__init__()
        self.d_model = d_model
        self.max_exercises = max_exercises
        self.vocab_size = vocab_size
        self.profile_embedding = nn.Linear(input_dim, d_model)
        self.exercise_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, max_exercises * vocab_size)

    def forward(self, src_profile, trg_sequence):
        memory = self.profile_embedding(src_profile).unsqueeze(1).repeat(1, 30, 1)
        trg_flat = trg_sequence.view(-1, self.max_exercises)
        embedded_trg = self.exercise_embedding(trg_flat).mean(dim=1)
        embedded_trg = embedded_trg.view(src_profile.shape[0], 30, self.d_model)
        embedded_trg = self.pos_encoder(embedded_trg)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(trg_sequence.size(1)).to(device)
        transformer_output = self.transformer_decoder(tgt=embedded_trg, memory=memory, tgt_mask=tgt_mask)
        output = self.fc_out(transformer_output)
        output = output.view(src_profile.shape[0], 30, self.max_exercises, self.vocab_size)
        return output

# --- Load the saved artifacts ---
production_scaler = joblib.load('fast_api_app/scaler.pkl')
production_encoder = joblib.load('fast_api_app/encoder.pkl')
production_tokenizer = PyTorchTokenizer.load_tokenizer('fast_api_app/tokenizer.json')
vocab_size = production_tokenizer.vocab_size

# Define model parameters (must match the trained model)
INPUT_DIM = 18 # 9 numerical features + 9 one-hot encoded features (2+4+3)
D_MODEL = 512
NHEAD = 8
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
MAX_EXERCISES_PER_DAY = 20

# Initialize and load the model
production_model = WorkoutGenerationTransformer(
    INPUT_DIM, D_MODEL, NHEAD, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, vocab_size, MAX_EXERCISES_PER_DAY
).to(device)
production_model.load_state_dict(torch.load('fast_api_app/trainium_sota_transformer_model.pth', map_location=device, weights_only=True))
production_model.eval()

print("✅ All production artifacts loaded successfully.")

# --- 2. Define the Main Prediction Function ---
def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Performs Top-p (nucleus) sampling on the logits distribution.
    """
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
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def generate_workout_plan(age, gender, goal, level, height, weight, chest, waist, hip, thigh, bicep):
    """
    This function takes all the inputs from the Gradio UI, preprocesses them,
    and returns a formatted workout plan using Top-p sampling and rule-based post-processing.
    """
    # --- a. Collate inputs ---
    bmi = round(weight / ((height / 100) ** 2), 1)
    user_profile = {
        'Age': age, 'Gender': gender, 'Goal': goal, 'level': level, 'height_cm': height,
        'weight_kg': weight, 'BMI': bmi, 'chest_cm': chest, 'waist_cm': waist,
        'hip_cm': hip, 'thigh_cm': thigh, 'bicep_cm': bicep
    }
    
    NUMERICAL_COLS = ['Age', 'height_cm', 'weight_kg', 'BMI', 'chest_cm', 'waist_cm', 'hip_cm', 'thigh_cm', 'bicep_cm']
    CATEGORICAL_COLS = ['Gender', 'Goal', 'level']
    user_df = pd.DataFrame([user_profile])
    
    # --- b. Preprocess the data ---
    user_cat = production_encoder.transform(user_df[CATEGORICAL_COLS])
    user_num = production_scaler.transform(user_df[NUMERICAL_COLS])
    user_processed = np.concatenate([user_num, user_cat], axis=1)
    src_tensor = torch.tensor(user_processed, dtype=torch.float32).to(device)
    
    # --- c. Run model inference with Top-p Sampling ---
    generated_sequence = torch.zeros((1, 30, MAX_EXERCISES_PER_DAY), dtype=torch.long).to(device)
    
    with torch.no_grad():
        for i in range(30):
            output_logits = production_model(src_tensor, generated_sequence)
            day_logits = output_logits[:, i, :, :]
            day_predictions = []
            for j in range(MAX_EXERCISES_PER_DAY):
                exercise_logits = day_logits[:, j, :]
                predicted_token = top_p_sampling(exercise_logits, p=0.92, temperature=1.0)
                day_predictions.append(predicted_token.item())
            generated_sequence[:, i, :] = torch.tensor(day_predictions, device=device).long()
    
    predicted_plan_ids = generated_sequence.cpu().numpy()
    
    # --- d. Decode and format the output with Rule-Based Post-Processing ---
    plan_dict = {"Day": [], "Workout": []}
    rest_day_string = "Rest Day"

    for i in range(30):
        day_number = i + 1
        plan_dict["Day"].append(f"Day {day_number}")
        
        # --- NEW: Enforce hard rule for rest days ---
        if day_number % 7 == 0:
            final_workout = rest_day_string
        else:
            # For workout days, generate exercises and clean them up
            day_tokens = predicted_plan_ids[0, i, :]
            raw_exercises = [production_tokenizer.idx_to_word.get(token, '') for token in day_tokens if token != 0]
            
            # Filter out the "Rest Day" token, then remove duplicates while preserving order
            unique_exercises = list(dict.fromkeys(
                ex for ex in raw_exercises if ex and ex != rest_day_string
            ))
            final_workout = ", ".join(unique_exercises)
            
        plan_dict["Workout"].append(final_workout)
        
    return pd.DataFrame(plan_dict)

# --- 3. Create and Launch the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Trainium AI Coach") as demo:
    gr.Markdown("# 🏋️ Trainium: AI Workout Plan Generator")
    gr.Markdown("Enter your profile details below to generate a hyper-personalized 30-day workout plan.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Bio-Data")
            age = gr.Slider(label="Age", minimum=16, maximum=80, value=25, step=1)
            gender = gr.Dropdown(label="Gender", choices=["Male", "Female"], value="Male")
            height = gr.Slider(label="Height (cm)", minimum=140, maximum=220, value=175, step=1)
            weight = gr.Slider(label="Weight (kg)", minimum=40, maximum=150, value=75, step=1)

            gr.Markdown("### Body Metrics (cm)")
            chest = gr.Number(label="Chest Circumference", value=100)
            waist = gr.Number(label="Waist Circumference", value=85)
            hip = gr.Number(label="Hip Circumference", value=95)
            thigh = gr.Number(label="Thigh Circumference", value=55)
            bicep = gr.Number(label="Bicep Circumference", value=35)
            
        with gr.Column(scale=2):
            gr.Markdown("### Fitness Goals")
            goal = gr.Dropdown(label="Primary Goal", choices=["Lose Weight", "Gain Muscle", "Gain Stamina", "General Fitness"], value="General Fitness")
            level = gr.Dropdown(label="Fitness Level", choices=["Beginner", "Intermediate", "Advanced"], value="Intermediate")
            
            btn = gr.Button("Generate My Plan", variant="primary", size="lg")
            
            gr.Markdown("## Your 30-Day Plan")
            output_plan = gr.Dataframe(
                headers=["Day", "Workout"], 
                datatype=["str", "str"], 
                row_count=30, 
                col_count=(2, "fixed"),
                column_widths=["10%", "90%"]
            )

    btn.click(
        fn=generate_workout_plan,
        inputs=[age, gender, goal, level, height, weight, chest, waist, hip, thigh, bicep],
        outputs=output_plan
    )

if __name__ == "__main__":
    demo.launch(share=True)