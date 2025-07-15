import torch
from src.data.preprocess import load_ECG_dataset
from torch.utils.data import DataLoader, random_split

def get_dataloaders(valid_part=0.2, 
                    batch_size=128, 
                    num_workers=2,
                    raw_path='data/raw/physionet.org/files/ptb-xl/1.0.1/',
                    sampling_rate=100, 
                    reduced_dataset=None):
    train_set, test_set, classes, features_num = load_ECG_dataset()
    
    d_train, d_val = random_split(train_set, [1 - valid_part, valid_part])
    train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(d_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)    
    return train_loader, test_loader, val_loader, classes, features_num