import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utils import get_device
from src.data.loader import get_dataloaders
from src.utils.constants import REDUCED_DISEASES_LIST 
from src.models.base_model import BaseModel
from src.models.cnn_handcrafted import HandcraftedModel
from src.data.preprocess import ECG_Dataset
from src.training.evaluator import evaluate_model
import tqdm
from torch.utils.data import DataLoader
import json

def train_model(model, train_set=None, test_set=None, val_set=None, class_names=None,
                epochs=10, learning_rate=0.001, is_handcrafted=False, batch_size=128, 
                save_path="models/checkpoints", save_name="no_name_model", device=None):
    if device is None:
        device = get_device()
    if train_set is None or test_set is None or val_set is None or class_names is None:
        train_set, test_set, val_set, class_names, features_num = get_dataloaders()
    size = len(train_set)
    if not model is None and not is_handcrafted:
        model = BaseModel(in_channels=12, out_classes=len(REDUCED_DISEASES_LIST)).to(device)
    elif not model is None:
        handcrafted_size = next(iter(train_set))[1].shape[1]
        base_model = BaseModel(in_channels=12, out_classes=len(REDUCED_DISEASES_LIST)).to(device)
        model = HandcraftedModel(base_model=base_model, handcrafted_classes=handcrafted_size).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        model.train()
        running_loss = 0
        total = 0

        for X, X_handcrafted, y in tqdm.tqdm(train_set, desc=f'Epoch {i+1}/{epochs}'):
            X, y = X.to(device), y.to(device)
            pred = None
            if not is_handcrafted:
                pred = model(X)
            else:
                X_handcrafted = X_handcrafted.to(device)
                pred = model(X, X_handcrafted)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            total += y.size(0)

        model.eval()
        Q_val = 0
        count_val = 0
        for X, X_handcrafted, y in tqdm.tqdm(val_set, desc=f'validation'):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                pred = None
                if not is_handcrafted:
                    pred = model(X)
                else:
                    X_handcrafted = X_handcrafted.to(device)
                    pred = model(X, X_handcrafted)
                loss = loss_fn(pred, y)
                Q_val += loss.item()
                count_val += 1
    
        Q_val /= count_val    
        epoch_loss = running_loss / total
        print(f"Epoch {i+1}/{epochs} - Loss: {epoch_loss:.4f}, Valid: {Q_val:.4f}")

    real_save_path = os.path.join(save_path, save_name)
    if not is_handcrafted:
        real_save_path = os.path.join(save_path, save_name)
        torch.save(model.state_dict(), real_save_path+".pth")
    else:
        real_save_path = os.path.join(save_path, "handcrafted_"+save_name)
        torch.save(model.state_dict(), real_save_path+".pth")
    return model

if __name__ == '__main__':
    train, test, valid, names, features_name = get_dataloaders()
    model = train_model(train, test, valid, names, is_handcrafted=True, epochs=20, batch_size=128)
