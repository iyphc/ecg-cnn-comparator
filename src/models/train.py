from math import e
import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils import get_device
from src.data.loader import get_dataloaders
from src.data.constants import REDUCED_DISEASES_LIST 
from src.models.base_model import BaseModel
from src.models.cnn_handcrafted import HandcraftedModel



def train_model(model=None, train_load=None, test_load=None, val_load=None, class_names=None, features_num=None, epochs=10, learning_rate=0.001, is_handcrafted=False, batch_size=128, device=None):
    if device is None:
        device = get_device()
    if train_load is None or test_load is None or val_load is None or class_names is None or features_num is None:
        train_load, test_load, val_load, class_names, features_num = get_dataloaders(batch_size=batch_size)
    size = len(train_load)
    if model is None:
        if is_handcrafted:
            model = HandcraftedModel(in_channels=12, out_classes=len(REDUCED_DISEASES_LIST), handcrafted_classes=features_num).to(device)
        else:
            model = BaseModel(in_channels=12, out_classes=len(REDUCED_DISEASES_LIST)).to(device)
    
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        model.train()
        running_loss = 0
        total = 0

        for X, X_handcrafted, y in tqdm.tqdm(train_load, desc=f'Epoch {i+1}/{epochs}'):
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
        for X, X_handcrafted, y in tqdm.tqdm(val_load, desc=f'validation'):
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

    if not is_handcrafted:
        torch.save(model.state_dict(), 'CNN_ECG_detection.pth')
    else:
        torch.save(model.state_dict(), 'handcrafted_CNN_ECG_detection.pth')
    print("Training complete! Model saved")
    return model

if __name__ == '__main__':
    train_load, test_load, valid_load, class_names, features_name = get_dataloaders()
    out_classes = len(class_names)
    model = BaseModel(12, out_classes)
    train_model(model, train_load, test_load, valid_load, class_names, is_handcrafted=True, epochs=20, batch_size=128)
