import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.utils import get_device, save_model
from ..data.loader import get_dataloaders
from ..models.base_model import BaseModel
from ..models.cnn_handcrafted import HandcraftedModel

import numpy as np
import sklearn.metrics
from tqdm import tqdm


def calculate_thresholds(all_probs, all_true, score_fn=sklearn.metrics.f1_score):
    best_thresholds_list = []
    best_scores_list = []
    thresholds = np.linspace(0, 1, 50)
    n_classes = all_probs.shape[1]
    for class_idx in range(n_classes):
        y_true_binary = all_true[:, class_idx]
        unique_labels = np.unique(y_true_binary)
        if len(unique_labels) < 2:
            print(
                f"Класс {class_idx} содержит только один уникальный класс: {unique_labels}"
            )
            best_thresholds_list.append(0.5)
            best_scores_list.append(0.0)
            continue
        best_score = 0.0
        best_threshold = 0.5
        for t in thresholds:
            y_pred_binary = (all_probs[:, class_idx] > t).astype(int)
            score_tmp = score_fn(
                y_true_binary, y_pred_binary, average="binary", zero_division=0
            )
            if score_tmp > best_score:
                best_score = score_tmp
                best_threshold = t
        best_thresholds_list.append(best_threshold)
        best_scores_list.append(best_score)
    mean_score = np.mean(best_scores_list)
    return best_thresholds_list, mean_score


def validate(
    model,
    val_loader,
    device,
    is_handcrafted,
    score_fn=sklearn.metrics.f1_score,
    loss_fn=nn.BCEWithLogitsLoss(),
):
    model.eval()
    all_probs = list()
    all_true = list()
    running_val_loss = 0
    total = 0
    with torch.no_grad():
        for inputs, X_handcrafted, labels in tqdm(val_loader, desc="validation epoch"):
            inputs = inputs.to(device)
            X_handcrafted = X_handcrafted.to(device)
            labels = labels.to(device)
            if is_handcrafted:
                outputs = model(inputs, X_handcrafted)
            else:
                outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_true.extend(labels.cpu().numpy())
    all_probs = np.vstack(all_probs)
    all_true = np.array(all_true)
    best_thresholds, mean_f1 = calculate_thresholds(
        all_probs, all_true, score_fn=score_fn
    )
    val_loss = running_val_loss / total
    return best_thresholds, mean_f1, val_loss


def calculate_pos_weight(train_load, device):
    all_labels = []
    for _, _, label in train_load:
        all_labels.append(label)
    tens_labels = torch.cat(all_labels, dim=0)
    pos_cnt = tens_labels.sum(dim=0)
    neg_cnt = tens_labels.shape[0] - pos_cnt
    pos_weight = torch.where(pos_cnt > 0, neg_cnt / pos_cnt, torch.ones_like(pos_cnt))
    return pos_weight.to(device)


def get_loss_fn(pos_weight):
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch(model, train_load, loss_fn, optimizer, device, is_handcrafted):
    model.train()
    running_loss = 0
    total = 0
    for X, X_handcrafted, y in tqdm(train_load, desc="Train epoch"):
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
    return epoch_loss


def train_model(
    model,
    train_load=None,
    test_load=None,
    val_load=None,
    class_names=None,
    epochs=10,
    learning_rate=0.001,
    is_handcrafted=False,
    save_path="models/checkpoints",
    save_name="no_name_model.pth",
    device=None,
    score_fn=sklearn.metrics.f1_score,
):
    if device is None:
        device = get_device()
    if (
        train_load is None
        or test_load is None
        or val_load is None
        or class_names is None
    ):
        raise FileNotFoundError("Incorrect datasets received")
    model = model.to(device)
    pos_weight = calculate_pos_weight(train_load, device)
    loss_fn = get_loss_fn(pos_weight)
    optimizer = get_optimizer(model, learning_rate)
    best_score = 0
    for i in range(epochs):
        train_epoch_loss = train_one_epoch(
            model, train_load, loss_fn, optimizer, device, is_handcrafted
        )
        tmp_thresh, tmp_best_score, val_epoch_loss = validate(
            model, val_load, device, is_handcrafted, score_fn=score_fn
        )
        if tmp_best_score > best_score:
            model.threshold = tmp_thresh
            best_score = tmp_best_score
        print(
            f"Epoch {i+1}/{epochs} - Train loss: {train_epoch_loss:.4f} / Val loss: {val_epoch_loss:.4f}"
        )
    real_save_path = os.path.join(save_path, save_name)
    save_model(model, real_save_path)
    print("Training complete! Model saved")
    return model
