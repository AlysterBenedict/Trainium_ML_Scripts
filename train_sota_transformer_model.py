import json
import time
import math
import re

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm # For the progress bar

# --- GPU Verification (Define device globally) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ### 2. Data Loading & Preprocessing (Pure PyTorch / Python)

# --- Custom Tokenizer and Padding Functions ---

class PyTorchTokenizer:
    """A custom tokenizer that mimics basic Keras Tokenizer functionality."""
    def __init__(self, oov_token='<unk>'):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.oov_token = oov_token
        self.word_to_idx['<pad>'] = 0
        self.idx_to_word[0] = '<pad>'
        self.word_to_idx[oov_token] = 1
        self.idx_to_word[1] = oov_token
        self.vocab_size = 2

    def fit_on_texts(self, corpus_iterator):
        unique_words = set()
        for text in corpus_iterator:
            words = [word.strip() for word in re.split(r',\s*', str(text))]
            unique_words.update(words)
        
        for word in sorted(list(unique_words)):
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            words = [word.strip() for word in re.split(r',\s*', str(text))]
            for word in words:
                seq.append(self.word_to_idx.get(word, self.word_to_idx[self.oov_token]))
            sequences.append(seq)
        return sequences

    def save_tokenizer(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word_to_idx, f, ensure_ascii=False, indent=4)

    @classmethod
    def load_tokenizer(cls, path):
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer.word_to_idx = json.load(f)
        tokenizer.idx_to_word = {i: w for w, i in tokenizer.word_to_idx.items()}
        tokenizer.vocab_size = len(tokenizer.word_to_idx)
        return tokenizer

def pad_sequences_numpy(sequences, maxlen, padding='post', truncating='post', value=0):
    """Pads sequences to the same length."""
    padded_sequences = np.full((len(sequences), maxlen), value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        if len(seq) > maxlen:
            if truncating == 'post':
                truncated = seq[:maxlen]
            else:
                truncated = seq[-maxlen:]
            padded_sequences[i] = np.array(truncated)
        else:
            if padding == 'post':
                padded_sequences[i, :len(seq)] = np.array(seq)
            else:
                padded_sequences[i, -len(seq):] = np.array(seq)
    return padded_sequences

# --- Main Data Loading Function ---
def load_and_preprocess_data():
    """Loads and preprocesses the entire dataset."""
    df = pd.read_csv('trainium_production_dataset_100k.csv')

    CATEGORICAL_COLS = ['Gender', 'Goal', 'level']
    NUMERICAL_COLS = ['Age', 'height_cm', 'weight_kg', 'BMI', 'chest_cm', 'waist_cm', 'hip_cm', 'thigh_cm', 'bicep_cm']
    OUTPUT_COLS = [f'Day_{i}' for i in range(1, 31)]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[CATEGORICAL_COLS])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[NUMERICAL_COLS])
    X = np.concatenate([scaled_features, encoded_features], axis=1)

    # MEMORY OPTIMIZATION: Create an iterator instead of a giant flat array
    def corpus_iterator():
        for col in OUTPUT_COLS:
            for text in df[col]:
                yield text

    tokenizer = PyTorchTokenizer()
    tokenizer.fit_on_texts(corpus_iterator())
    
    y_sequences = [tokenizer.texts_to_sequences(df[col].tolist()) for col in OUTPUT_COLS]
    y_padded = [pad_sequences_numpy(seq, maxlen=20) for seq in y_sequences]
    y = np.stack(y_padded, axis=1)

    return X, y, scaler, encoder, tokenizer

# ### 3. SOTA Model Architecture: Decoder-only Transformer

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

# ### 4. Training with AMP & Gradient Accumulation

# --- SOTA Training and Evaluation Functions ---
def train(model, iterator, optimizer, criterion, clip, scaler, accumulation_steps=4):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(iterator, desc="Training", leave=False)

    for i, batch in enumerate(progress_bar):
        src = batch[0].to(device)
        trg = batch[1].to(device)
        
        with torch.amp.autocast(device_type="cuda"):
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg.view(-1)
            loss = criterion(output, trg) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        epoch_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({'loss': epoch_loss / (i + 1)})

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg.view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Print device info ONCE ---
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # --- 1. Load Data ---
    print("\nLoading and preprocessing data...")
    X, y, scaler, encoder, tokenizer = load_and_preprocess_data()
    vocab_size = tokenizer.vocab_size
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    BATCH_SIZE = 128
    NUM_WORKERS = 22
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("\n--- Data Preprocessing Summary ---")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"CPU Workers for DataLoader: {NUM_WORKERS}")
    
    # --- 2. Initialize Model and Training Components ---
    INPUT_DIM = X_train.shape[1]
    D_MODEL = 512
    NHEAD = 8
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    MAX_EXERCISES_PER_DAY = 20

    model = WorkoutGenerationTransformer(
        INPUT_DIM, D_MODEL, NHEAD, NUM_DECODER_LAYERS, 
        DIM_FEEDFORWARD, vocab_size, MAX_EXERCISES_PER_DAY
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    grad_scaler = torch.amp.GradScaler()

    # --- 3. Run Training Loop ---
    N_EPOCHS = 20
    CLIP = 1
    ACCUMULATION_STEPS = 4
    best_valid_loss = float('inf')

    print(f"\n--- Starting SOTA Transformer Training ---")
    print(f"Simulated Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, grad_scaler, ACCUMULATION_STEPS)
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'trainium_sota_transformer_model.pth')
        
        print(f'Epoch: {epoch+1:02} | Time: {int(end_time - start_time)}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        
    # --- 4. Save Artifacts Post-Training ---
    print("\n--- Saving Final Artifacts ---")
    print("✅ SOTA PyTorch Transformer model saved as 'trainium_sota_transformer_model.pth'")
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Scaler saved as 'scaler.pkl'")
    joblib.dump(encoder, 'encoder.pkl')
    print("✅ Encoder saved as 'encoder.pkl'")
    tokenizer.save_tokenizer('tokenizer.json')
    print("✅ Custom Tokenizer saved as 'tokenizer.json'")