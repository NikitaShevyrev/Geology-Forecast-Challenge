# Final training and submission script for Geology Forecast Challenge

import numpy as np
import pandas as pd
import random
import os
from os.path import basename
from pathlib import Path
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
import torch.optim as optim


from data import process_raw_file
from utils import init_wandb
from augmentations import mixup
from data import GeologyDataset
from model import ParallelLSTMWithAttention
from training import train_model_with_nll_loss, validate_model
from metrics import compute_nll_score


# 🔹 Reproducibility
def seed_everything(seed=2025):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SEED = 2025
seed_everything(SEED)


# ----------------------
# CONFIGURATION
# ----------------------
RUN_DATA_GENERATION = False   # Set to True if regenerating synthetic data
SYNTHETIC_CSV = "input/geology-forecast-challenge-open/data/synthetic_train.csv"
TRAIN_CSV = "input/geology-forecast-challenge-open/data/train.csv"
TEST_CSV = "input/geology-forecast-challenge-open/data/test.csv"
SAMPLE_SUBMISSION_CSV = "input/geology-forecast-challenge-open/data/sample_submission.csv"
WEIGHTS_DIR = "weights"  # contains model_fold_{i}.pt
OUTPUT_SUBMISSION = "submission.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not Path(WEIGHTS_DIR).exists():
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)


# ----------------------
# DATA GENERATION
# ----------------------

if RUN_DATA_GENERATION:
    all_files = glob.glob("./input/geology-forecast-challenge-open/data/train_raw/*.csv")
    all_data = []
    for f in tqdm(all_files):
        all_data.extend(process_raw_file(f, max_chunks=38190//len(all_files)+1))
    df = pd.DataFrame(all_data)
    df.to_csv(SYNTHETIC_CSV, index=False)
    print(f"✅ Saved {len(df)} synthetic samples to {basename(SYNTHETIC_CSV)}")


# ----------------------
# DATA PREPARATION
# ----------------------
synth_train_df = pd.read_csv(SYNTHETIC_CSV)
synth_train_df = synth_train_df.iloc[:38190,:]

train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df.iloc[:,:601]

train_df = pd.concat([train_df, synth_train_df], ignore_index=True)

submission_template = pd.read_csv(SAMPLE_SUBMISSION_CSV)

REALIZATIONS = [col for col in submission_template.columns if col != "geology_id"]
NUM_REALIZATIONS = len(REALIZATIONS) // 300

print(f"🔍 Detected {NUM_REALIZATIONS} realizations.")


columns_new = ['geology_id']
for r in range(NUM_REALIZATIONS):
    for i in range(-299,301):
        columns_new.append(f'r_{r}_pos_{i}')

train_df_new = pd.DataFrame(
    data=np.concatenate((np.full((train_df.shape[0]//10,1), 'none'), train_df.iloc[:,1:].values.reshape((-1,10*600))), axis=1),
    columns=columns_new
)

train_FEATURES = []
for r in range(NUM_REALIZATIONS):
    for i in range(-299,1):
        train_FEATURES.append(f'r_{r}_pos_{i}')

train_REALIZATIONS = []
for r in range(NUM_REALIZATIONS):
    for i in range(1,301):
        train_REALIZATIONS.append(f'r_{r}_pos_{i}')

train_df = train_df_new
train_df = train_df.iloc[:,1:]
train_df = train_df.astype('float64')
train_df.insert(0, 'geology_id', 'none')

# 🔧 Replacing nans
train_df.iloc[:, 1:] = train_df.iloc[:, 1:].fillna(0)


# ----------------------
# Main Orchestrator
# ----------------------
def train_and_predict(
    fold_idx, 
    train_index, 
    val_index, 
    X_num, 
    y,
    config,
    oof_df=None
):
    fold_config = config.copy()
    fold_config.update({"fold": fold_idx})
    
    run = init_wandb(config=fold_config)
    
    # Train data (EXPANDED across 10 realizations)
    X_num_train = X_num[train_index].reshape((-1,300))
    y_train = y[train_index].reshape((-1,300))
    realization_ids_train = np.tile(np.arange(NUM_REALIZATIONS), len(train_index))
    
    # Validation data (USE ONLY REALIZATION 0)
    X_num_val = X_num[val_index][:, :300]
    y_val = y[val_index][:, :300]
    realization_ids_val = np.zeros(len(X_num_val), dtype=int)

    X_num_train, y_train = mixup(X_num_train, y_train, 0.4)
    
    train_dataset = GeologyDataset(X_num_train, y_train, realization_ids_train)
    val_dataset = GeologyDataset(X_num_val, y_val, realization_ids_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        pin_memory=True, 
        num_workers=2  
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    model = ParallelLSTMWithAttention(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=300,
        dropout=config['dropout'],
        num_realizations=NUM_REALIZATIONS,
        realization_emb_dim=16,
        use_multihead=True,
        num_heads=2,
        fusion_method='concat'  # 'concat' or 'add', 'gated'
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8  # Increased stability
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,       
        T_mult=2,   
        eta_min=1e-6 
    )
    
    best_val_loss = float('inf')
    val_predictions = np.zeros((len(val_index), len(REALIZATIONS)))

    print(f"Training fold {fold_idx + 1}...")
    for epoch in range(config['epochs']):
        train_loss = train_model_with_nll_loss(model, train_loader, optimizer, device)
        
        val_loss, val_preds, val_targets = validate_model(model, val_loader, device)
        
        val_predictions = val_preds
        
        scheduler.step()

        if run:
            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"{WEIGHTS_DIR}/model_fold_{fold_idx}.pt"
            torch.save(model.state_dict(), model_path)
            if run:
                run.save(model_path)
    
    oof_df.loc[val_index, REALIZATIONS[:300]] = val_predictions
    
    if run:
        run.finish()
    
    return oof_df


# ----------------------
# Training Loop
# ----------------------
config = {
    'model_type': 'LSTM',
    'hidden_size': 1024,
    'num_layers': 3,
    'dropout': 0.2,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'batch_size': 256,
    'epochs': 30,
    'seed': SEED,
}

oof_df = train_df[['geology_id'] + train_REALIZATIONS].copy() # out-of-fold predictions
solution = train_df[['geology_id'] + train_REALIZATIONS].copy()

folds = 5
kf = KFold(n_splits=folds, random_state=SEED, shuffle=True)

X_num, y = train_df[train_FEATURES].values, train_df[train_REALIZATIONS].values # NOTE: We take one realization only train_df.iloc[:,301:601].values

val_scores = []

for fold_idx, (train_index, val_index) in enumerate(kf.split(X_num)):
    oof_df = train_and_predict(
        fold_idx, 
        train_index, 
        val_index, 
        X_num, 
        y,
        config,
        oof_df
    )
    
    fold_val_preds = oof_df.loc[val_index, ['geology_id'] + train_REALIZATIONS]
    fold_val_solution = solution.loc[val_index]
    
    fold_score = compute_nll_score(fold_val_solution, fold_val_preds)
    val_scores.append(fold_score)
    
    print(f"Fold {fold_idx+1} validation NLL score: {fold_score:.6f}")

avg_val_score = np.mean(val_scores)
print(f"Average validation NLL score: {avg_val_score:.6f}")
