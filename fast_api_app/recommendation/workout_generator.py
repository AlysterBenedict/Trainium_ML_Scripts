# workout_generator.py

import torch
import torch.nn as nn
import math
import joblib
import json
import pandas as pd
import numpy as np
import os

# --- Recommender Model Architecture ---
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
        # src_profile shape: [batch_size, input_dim]
        # trg_sequence shape: [batch_size, seq_len, max_exercises]
        
        # Create memory from the user profile, repeated for each day in the sequence
        memory = self.profile_embedding(src_profile).unsqueeze(1).repeat(1, trg_sequence.size(1), 1)
        
        # Embed the target exercise sequence
        trg_flat = trg_sequence.view(trg_sequence.size(0) * trg_sequence.size(1), self.max_exercises)
        embedded_trg = self.exercise_embedding(trg_flat).mean(dim=1)
        
        # Reshape using the actual sequence length from the input
        embedded_trg = embedded_trg.view(trg_sequence.size(0), trg_sequence.size(1), self.d_model)
        
        embedded_trg = self.pos_encoder(embedded_trg)
        
        # Generate a mask to prevent the model from looking ahead in the sequence
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(trg_sequence.size(1)).to(src_profile.device)
        
        # Get the transformer's output
        transformer_output = self.transformer_decoder(tgt=embedded_trg, memory=memory, tgt_mask=tgt_mask)
        output = self.fc_out(transformer_output)
        
        # Reshape the final output
        output = output.view(src_profile.shape[0], trg_sequence.size(1), self.max_exercises, self.vocab_size)
        return output


class PyTorchTokenizer:
    """A helper class to load the tokenizer config."""
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

    @classmethod
    def load_tokenizer(cls, path):
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer.word_to_idx = json.load(f)
        tokenizer.idx_to_word = {i: w for w, i in tokenizer.word_to_idx.items()}
        tokenizer.vocab_size = len(tokenizer.word_to_idx)
        return tokenizer

class WorkoutGenerator:
    def __init__(self, model_path, tokenizer_path, scaler_path, encoder_path, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = PyTorchTokenizer.load_tokenizer(tokenizer_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

        self.CATEGORICAL_COLS = ['Gender', 'Goal', 'level']
        self.NUMERICAL_COLS = ['Age', 'height_cm', 'weight_kg', 'BMI', 'chest_cm', 'waist_cm', 'hip_cm', 'thigh_cm', 'bicep_cm']

        INPUT_DIM = 18
        D_MODEL = 512
        NHEAD = 8
        NUM_DECODER_LAYERS = 6
        DIM_FEEDFORWARD = 2048
        self.MAX_EXERCISES_PER_DAY = 20

        self.model = WorkoutGenerationTransformer(
            INPUT_DIM, D_MODEL, NHEAD, NUM_DECODER_LAYERS,
            DIM_FEEDFORWARD, self.tokenizer.vocab_size, self.MAX_EXERCISES_PER_DAY
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def generate_workout_plan(self, user_profile):
        """
        Generates a 30-day workout plan using an auto-regressive approach with top-k sampling.
        """
        profile_df = pd.DataFrame([user_profile])
        encoded_features = self.encoder.transform(profile_df[self.CATEGORICAL_COLS])
        scaled_features = self.scaler.transform(profile_df[self.NUMERICAL_COLS])
        processed_profile = np.concatenate([scaled_features, encoded_features], axis=1)

        profile_tensor = torch.tensor(processed_profile, dtype=torch.float32).to(self.device)

        generated_sequence = torch.zeros((1, 1, self.MAX_EXERCISES_PER_DAY), dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(29): # Generate for the next 29 days to get a total of 30
                output_logits = self.model(profile_tensor, generated_sequence)
                last_day_logits = output_logits[:, -1, :, :]

                # --- Top-k sampling to prevent repetition ---
                k = 10 
                top_k_logits, top_k_indices = torch.topk(last_day_logits, k, dim=-1)
                
                # Replace logits outside the top-k with -infinity
                filtered_logits = torch.full_like(last_day_logits, float('-inf'))
                filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Get probabilities using softmax
                probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
                
                # Sample from the probability distribution
                probabilities_2d = probabilities.squeeze(0)
                predicted_ids_for_next_day = torch.multinomial(probabilities_2d, 1).squeeze(1)
                
                # Reshape back for concatenation
                predicted_ids_for_next_day = predicted_ids_for_next_day.unsqueeze(0)
                
                generated_sequence = torch.cat([generated_sequence, predicted_ids_for_next_day.unsqueeze(1)], dim=1)

        # Decode the final generated sequence
        workout_plan = {}
        full_plan_ids = generated_sequence.squeeze(0).cpu().numpy()
        for day_idx in range(full_plan_ids.shape[0]):
            day_exercises = []
            for exercise_id in full_plan_ids[day_idx]:
                if exercise_id == 0:
                    break
                exercise_name = self.tokenizer.idx_to_word.get(exercise_id, "<unk>")
                
                # Filter out unwanted tokens, "Rest Day", and duplicates
                if exercise_name not in ["<unk>", "<pad>", "Rest Day"] and exercise_name not in day_exercises:
                    day_exercises.append(exercise_name)
            
            if day_exercises:
                day_key = f"Day_{day_idx + 1}"
                workout_plan[day_key] = day_exercises

        # Manually insert a rest day every 7th day, replacing any generated workout
        for day_idx in range(1, 31):
            if day_idx % 7 == 0:
                day_key = f"Day_{day_idx}"
                workout_plan[day_key] = ["Rest Day"]

        return workout_plan

