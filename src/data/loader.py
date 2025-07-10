import torch
from src.data.preprocess import load_ECG_dataset
from torch.utils.data import DataLoader, random_split

def get_dataloaders(valid_part=0.2, batch_size=128, num_workers=2):
    train_set, test_set, classes = load_ECG_dataset()
    
    d_train, d_val = random_split(train_set, [1 - valid_part, valid_part])
    train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(d_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)    
    return train_loader, test_loader, val_loader, classes