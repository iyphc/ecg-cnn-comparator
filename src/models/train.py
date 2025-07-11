import tqdm
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils import load_model, save_model
from sklearn.metrics import f1_score
from src.models.utils import get_device
from src.data.loader import get_dataloaders
from src.data.constants import REDUCED_DISEASES_LIST 
from src.models.base_model import BaseModel
from src.models.cnn_handcrafted import HandcraftedModel

def validate(model, val_loader, device, loss_fn, is_handcrafted=False):
    model.eval()
    all_probs = []
    all_true = []

    Q_val = 0
    count_val = 0
    with torch.no_grad():
        for inputs, X_handcrafted, labels in tqdm.tqdm(val_loader, desc=f'valudate'):
            inputs = inputs.to(device)
            X_handcrafted = X_handcrafted.to(device)
            labels = labels.to(device)

            if is_handcrafted:
                outputs = model(inputs, X_handcrafted)
            else:
                outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            Q_val += loss.item()
            count_val += 1
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_true.extend(labels.cpu().numpy())

    Q_val /= count_val    
    print(f"Val: {Q_val:.4}")

    best_f1 = 0
    best_threshold = 0.5
    thresholds = np.linspace(0, 1, 50)

    for t in thresholds:
        preds = (np.array(all_probs) > t).astype(int)
        f1 = f1_score(all_true, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    return best_threshold, best_f1

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
    
    best_f1 = 0

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

        epoch_loss = running_loss / total
        tmp_thresh, tmp_best_f1 = validate(model, val_load, device, loss_fn, is_handcrafted)
        if tmp_best_f1 > best_f1:
            model.threshold = tmp_thresh
        print(f"Epoch {i+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    if not is_handcrafted:
        save_model(model, 'CNN_ECG_detection.pth')
    else:
        save_model(model, 'handcrafted_CNN_ECG_detection.pth')
    print("Training complete! Model saved")
    print(f"THRESHOLD: {model.threshold}")
    return model

if __name__ == '__main__':
    train_load, test_load, valid_load, class_names, features_name = get_dataloaders()
    out_classes = len(class_names)
    model = BaseModel(12, out_classes)
    train_model(model, train_load, test_load, valid_load, class_names, is_handcrafted=False, epochs=20, batch_size=128)
