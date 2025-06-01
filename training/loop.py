import torch
import torch.nn.functional as F
import numpy as np


def train_model_with_nll_loss(model, train_loader, optimizer, device):
    model.train()
    train_losses = []
    
    for data, target, rid in train_loader:
        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        rid = rid.to(device)
        output = model(data, rid)
        
        target_mean = target.mean(dim=0)
        target_std = target.std(dim=0) + 1e-6

        normalized_output = (output - target_mean) / target_std
        normalized_target = (target - target_mean) / target_std

        loss = F.mse_loss(normalized_output, normalized_target)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_losses.append(loss.item())
    
    return np.mean(train_losses)

def validate_model(model, val_loader, device):
    model.eval()
    val_losses = []
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for data, target, rid in val_loader:
            data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
            rid = rid.to(device)
            output = model(data, rid)
            
            target_mean = target.mean(dim=0)
            target_std = target.std(dim=0) + 1e-6

            normalized_output = (output - target_mean) / target_std
            normalized_target = (target - target_mean) / target_std
            
            loss = F.mse_loss(normalized_output, normalized_target)
            
            val_losses.append(loss.item())
            val_preds.append(output.cpu().numpy())
            val_targets.append(target.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    
    return np.mean(val_losses), val_preds, val_targets