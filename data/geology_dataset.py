from torch.utils.data import Dataset

class GeologyDataset(Dataset):
    def __init__(self, features, targets=None, realization_ids=None, is_test=False):
        self.features = features
        self.targets = targets
        self.realization_ids = realization_ids
        self.is_test = is_test
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx].reshape(-1, 1)  # [50, 1]

        if self.is_test:
            return x
        else:
            y = self.targets[idx]  # [300]
            rid = self.realization_ids[idx]  # scalar
            return x, y, rid