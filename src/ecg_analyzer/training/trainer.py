import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.utils import get_device, save_model
from ..data.loader import get_dataloaders
from ..models.base_model import BaseModel
from ..models.cnn_handcrafted import HandcraftedModel

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch


def validate(model, val_loader, device, is_handcrafted):
    model.eval()
    all_probs = []
    all_true = []

    with torch.no_grad():
        for inputs, X_handcrafted, labels in tqdm(val_loader, desc="validate"):
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

    mean_f1 = 0

    for class_idx in range(n_classes):
        y_true_binary = all_true[:, class_idx]

        unique_labels = np.unique(y_true_binary)
        if len(unique_labels) < 2:
            print(
                f"Класс {class_idx} содержит только один уникальный класс: {unique_labels}"
            )
            best_thresholds.append(0.5)
            best_f1_scores.append(0.0)
            continue

        best_f1 = 0.0
        best_threshold = 0.5

        for t in thresholds:
            y_pred_binary = (all_probs[:, class_idx] > t).astype(int)
            f1 = f1_score(
                y_true_binary, y_pred_binary, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)
        mean_f1 = np.mean(best_f1_scores)

    return best_thresholds, mean_f1


def train_model(
    model,
    train_load=None,
    test_load=None,
    val_load=None,
    class_names=None,
    epochs=10,
    learning_rate=0.001,
    val_part=0.2,
    is_handcrafted=False,
    handcrafted_size=0,
    batch_size=128,
    save_path="models/checkpoints",
    save_name="no_name_model.pth",
    device=None,
    num_workers=2,
    sampling_rate=100,
    reduced_dataset=None,
    features=None,
):

    if device is None:
        device = get_device()

    if (
        train_load is None
        or test_load is None
        or val_load is None
        or class_names is None
    ):
        print("THE DATA IS EMPTY")
        return

    model = model.to(device)

    pos_weight = None
    all_labels = []
    for _, _, label in train_load:
        all_labels.append(label)

    tens_labels = torch.cat(all_labels, dim=0)
    pos_cnt = tens_labels.sum(dim=0)

    neg_cnt = tens_labels.shape[0] - pos_cnt

    pos_weight = torch.where(pos_cnt > 0, neg_cnt / pos_cnt, torch.ones_like(pos_cnt))
    pos_weight = pos_weight.to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0

    for i in range(epochs):
        model.train()
        running_loss = 0
        total = 0

        for X, X_handcrafted, y in tqdm(train_load, desc=f"Epoch {i+1}/{epochs}"):
            X, y = X.to(device), y.to(device)
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

    real_save_path = os.path.join(save_path, save_name)
    save_model(model, real_save_path)

    print("Training complete! Model saved")
    return model
