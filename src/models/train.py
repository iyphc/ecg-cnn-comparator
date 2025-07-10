import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils import get_device, get_dataloaders
from src.data.constants import REDUCED_DISEASES_LIST 
from src.models.base_model import BaseModel
from src.models.cnn_handcrafted import HandcraftedModel
from src.data.preprocess import ECG_Dataset
from src.models.evaluation import evaluate_model
import tqdm
from torch.utils.data import DataLoader

def train_model(model, train_set=None, test_set=None, val_set=None, class_names=None, epochs=10, learning_rate=0.001, is_handcrafted=False, batch_size=128, device=None):
    if device is None:
        device = get_device()
    if train_set is None or test_set is None or val_set is None or class_names is None:
        train_set, test_set, val_set, class_names = get_dataloaders()
    size = len(train_set)
    if not model is None and not is_handcrafted:
        model = BaseModel(in_channels=12, out_classes=len(REDUCED_DISEASES_LIST)).to(device)
    elif not model is None:
        handcrafted_size = next(iter(train_set))[1].shape[1]
        model = HandcraftedModel(in_channels=12, out_classes=len(REDUCED_DISEASES_LIST), handcrafted_classes=handcrafted_size).to(device)

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

    if not is_handcrafted:
        torch.save(model.state_dict(), 'CNN_ECG_detection.pth')
    else:
        torch.save(model.state_dict(), 'handcrafted_CNN_ECG_detection.pth')
    scores = evaluate_model(model, test_set, is_handcrafted, device=None)
    print(f"Test F1: {scores["f1_None"]}")
    print("Training complete! Model saved")
    return model

if __name__ == '__main__':
    train, test, valid, names = get_dataloaders()
    train_model(train, test, valid, names, is_handcrafted=True, epochs=20, batch_size=128)
