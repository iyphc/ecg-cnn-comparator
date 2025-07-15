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

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch

def validate(model, val_loader, device, is_handcrafted):
    model.eval()
    all_probs = []
    all_true = []

    with torch.no_grad():
        for inputs, X_handcrafted, labels in tqdm(val_loader, desc='validate'):
            inputs = inputs.to(device)
            X_handcrafted = X_handcrafted.to(device)
            labels = labels.to(device)
            if is_handcrafted:
                outputs = model(inputs, X_handcrafted)
            else:
                outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_true.extend(labels.cpu().numpy())
    all_probs = np.vstack(all_probs)
    all_true = np.array(all_true)

    best_thresholds = []
    best_f1_scores = []

    thresholds = np.linspace(0, 1, 50)
    n_classes = all_probs.shape[1]

    for class_idx in range(n_classes):
        y_true_binary = all_true[:, class_idx]

        unique_labels = np.unique(y_true_binary)
        if len(unique_labels) < 2:
            print(f"Класс {class_idx} содержит только один уникальный класс: {unique_labels}")
            best_thresholds.append(0.5)
            best_f1_scores.append(0.0)
            continue

        best_f1 = 0.0
        best_threshold = 0.5

        for t in thresholds:
            y_pred_binary = (all_probs[:, class_idx] > t).astype(int)
            f1 = f1_score(y_true_binary, y_pred_binary, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)
        mean_f1 = np.mean(best_f1_scores)

    return best_thresholds, mean_f1

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

        for X, X_handcrafted, y in tqdm(train_load, desc=f'Epoch {i+1}/{epochs}'):
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
        tmp_thresh, tmp_best_f1 = validate(model, val_load, device, is_handcrafted)
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
