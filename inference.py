# Inference script using pretrained weights to generate submission.csv

import numpy as np
import pandas as pd
import os
import random
import torch
from torch.utils.data import DataLoader


from data import GeologyDataset
from model import ParallelLSTMWithAttention


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
TEST_CSV = "input/geology-forecast-challenge-open/data/test.csv"
SAMPLE_SUBMISSION_CSV = "input/geology-forecast-challenge-open/data/sample_submission.csv"
WEIGHTS_DIR = "weights"
OUTPUT_SUBMISSION = "submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 2025
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
FOLDS = 5


# ----------------------
# Load Test Data
# ----------------------
test_df = pd.read_csv(TEST_CSV)
submission_template = pd.read_csv(SAMPLE_SUBMISSION_CSV)

FEATURES = [col for col in test_df.columns if col != "geology_id"]
REALIZATIONS = [col for col in submission_template.columns if col != "geology_id"]
NUM_REALIZATIONS = len(REALIZATIONS) // 300


# ----------------------
# Run Inference
# ----------------------
X_num_test = test_df[FEATURES].values
test_dataset = GeologyDataset(X_num_test, is_test=True)
test_loader = DataLoader(
    test_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False,
    pin_memory=True,
    num_workers=2
)

test_preds_all_folds = np.zeros((FOLDS, len(test_df), len(REALIZATIONS)))

for fold in range(FOLDS):
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
    model.load_state_dict(
        torch.load(f"{WEIGHTS_DIR}/model_fold_{fold}.pt", map_location=device)
    )
    model.eval()
    test_predictions = np.zeros((len(X_num_test), len(REALIZATIONS)))
    test_preds = []    
    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]
            data = data.to(device, dtype=torch.float32)
    
            outputs = []
            for r_id in range(NUM_REALIZATIONS):
                realization_ids = torch.full((data.size(0),), r_id, dtype=torch.long, device=device)
                preds = model(data, realization_ids)  # [B, 300]
                outputs.append(preds.cpu().numpy())   # Append [B, 300]
    
            # Stack into shape: [B, 10, 300] → then reshape to [B, 3000]
            outputs = np.stack(outputs, axis=1).reshape(data.size(0), 300 * NUM_REALIZATIONS)
            test_preds.append(outputs)
    
    test_predictions = np.concatenate(test_preds)
    test_preds_all_folds[fold] = test_predictions


# ----------------------
# Save Submission
# ----------------------
test_preds_avg = np.mean(test_preds_all_folds, axis=0)
submission = submission_template.copy()
submission[REALIZATIONS] = test_preds_avg
submission.to_csv('submission.csv', index=False)
print("✅ Submission saved as submission.csv")
