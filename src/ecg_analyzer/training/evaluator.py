from src.ecg_analyzer.data.loader import get_dataloaders
from src.ecg_analyzer.utils.utils import get_device
from src.ecg_analyzer.models.base_model import BaseModel
from src.ecg_analyzer.models.cnn_handcrafted import HandcraftedModel
import tqdm
import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from collections import defaultdict

# Deprecated
def confusion_matrix(all_true, all_pred):
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    matrix = {}
    matrix["TP"] = np.sum((all_true == 1) & (all_pred == 1), axis = 0)
    matrix["FP"] = np.sum((all_true == 0) & (all_pred == 1), axis = 0)
    matrix["FN"] = np.sum((all_true == 1) & (all_pred == 0), axis = 0)
    matrix["TN"] = np.sum((all_true == 0) & (all_pred == 0), axis = 0)
    return matrix

def basic_scores(all_true, all_pred, threshold=0.5):
    scores = {}

    all_pred_prob = np.array(all_pred)
    all_pred = np.array(all_pred) > threshold

    eps = 1e-8
    
    scores["recall_samples"] = recall_score(all_true, all_pred, average='samples', zero_division=0)
    scores["recall_micro"] = recall_score(all_true, all_pred, average='micro', zero_division=0)
    scores["recall_macro"] = recall_score(all_true, all_pred, average='macro', zero_division=0)
    scores["recall_weighted"] = recall_score(all_true, all_pred, average='weighted', zero_division=0)
    scores['recall_none'] = recall_score(all_true, all_pred, average=None, zero_division=0)
    scores["precision_samples"] = precision_score(all_true, all_pred, average='samples', zero_division=0)
    scores["precision_micro"] = precision_score(all_true, all_pred, average='micro', zero_division=0)
    scores["precision_macro"] = precision_score(all_true, all_pred, average='macro', zero_division=0)
    scores["precision_weighted"] = precision_score(all_true, all_pred, average='weighted', zero_division=0)
    scores['precision_none'] = precision_score(all_true, all_pred, average=None, zero_division=0)
    scores['f1_samples'] = f1_score(all_true, all_pred, average='samples', zero_division=0)
    scores['f1_micro'] = f1_score(all_true, all_pred, average='micro', zero_division=0)
    scores['f1_macro'] = f1_score(all_true, all_pred, average='macro', zero_division=0)
    scores['f1_weighted'] = f1_score(all_true, all_pred, average='weighted', zero_division=0)
    scores['f1_none'] = f1_score(all_true, all_pred, average=None, zero_division=0)
    scores['roc-auc-macro'] = roc_auc_score(all_true, all_pred_prob, multi_class='ovo', average='macro')
    scores['roc-auc-elems'] = roc_auc_score(all_true, all_pred_prob, average=None)

    return scores

def compare_models(base_model, handcrafted_model, test_loader, seed=52, repet_number=1000, alpha=0.05):
    np.random.seed(seed)
    base_model_pred, all_true = evaluate_model(base_model, test_loader)
    handcrafted_model_pred, _ = evaluate_model(handcrafted_model, test_loader, is_handcrafted=True)
    n_samples = len(all_true)
    
    delta_dict = defaultdict(list)

    all_true = np.array(all_true)
    base_model_pred = np.array(base_model_pred)
    handcrafted_model_pred = np.array(handcrafted_model_pred)

    for _ in range(repet_number):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        mixed_all_true = all_true[idx]
        mixed_base_pred = base_model_pred[idx]
        mixed_handcrafted_pred = handcrafted_model_pred[idx]

        base_score = basic_scores(mixed_all_true, mixed_base_pred)
        hand_score = basic_scores(mixed_all_true, mixed_handcrafted_pred)

        for name in base_score:
            diff = np.array(base_score[name]) - np.array(hand_score[name])
            delta_dict[name].append(diff)

    ci = {}
    for name, deltas in delta_dict.items():
        lower = np.percentile(deltas, 100 * (alpha / 2))
        upper = np.percentile(deltas, 100 * (1 - alpha / 2))
        ci[name] = [lower, upper]

    return ci

def evaluate_model(model, test_loader, is_handcrafted=False, device=None):
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()
    
    all_preds = []
    all_true = []
    flag = False
    with torch.no_grad():
        for X, X_handcrafted, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = None
            if (not is_handcrafted):
                outputs = model(X)
            else:
                X_handcrafted = X_handcrafted.to(device)
                outputs = model(X, X_handcrafted)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            true = y.cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(true)
    
    return all_preds, all_true

if __name__ == "__main__":
    is_handcrafted = True
    _, test, _, names, features = get_dataloaders()
    if is_handcrafted:
        base_model = BaseModel(in_channels=12, out_classes=len(names))
        model = HandcraftedModel(base_model=base_model, handcrafted_classes=len(features))
        state_dict = torch.load("handcrafted_CNN_ECG_detection.pth", weights_only=False)
        model.load_state_dict(state_dict['model_state_dict'])
        model.threshold = state_dict['threshold']
    else:
        model = BaseModel(in_channels=12, out_classes=len(names))
        state_dict = torch.load("CNN_ECG_detection.pth", weights_only=False)
        model.load_state_dict(state_dict['model_state_dict'])
        model.threshold = state_dict['threshold']

    print("\n\n---------")
    print("SCORES:")
    print("---------\n")

    base_model_pred, all_pred = evaluate_model(model=model, test_loader=test, is_handcrafted=is_handcrafted)
    scores = basic_scores(all_pred, base_model_pred, threshold=model.threshold)

    for name in scores:
        print(f"{name}: {scores[name]}")
    
