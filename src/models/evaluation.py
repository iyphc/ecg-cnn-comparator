from src.data.loader import get_dataloaders
from src.models.utils import get_device
from src.models.base_model import BaseModel
from src.models.cnn_handcrafted import HandcraftedModel
import tqdm
import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import f1_score, roc_auc_score
from collections import defaultdict

def confusion_matrix(all_true, all_pred):
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    matrix = {}
    matrix["TP"] = np.sum((all_true == 1) & (all_pred == 1), axis = 0)
    matrix["FP"] = np.sum((all_true == 0) & (all_pred == 1), axis = 0)
    matrix["FN"] = np.sum((all_true == 1) & (all_pred == 0), axis = 0)
    matrix["TN"] = np.sum((all_true == 0) & (all_pred == 0), axis = 0)

    return matrix

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


def basic_scores(all_true, all_pred):
    scores = {}

    matrix = confusion_matrix(all_true, all_pred)
    eps = 1e-8
    
    scores["sensitivity"] = matrix["TP"] / (matrix["TP"] + matrix["FN"] + eps)
    scores["specificity"] = matrix["TN"] / (matrix["TN"] + matrix["FP"] + eps)
    scores["mean_sensitivity"] = sum(scores["sensitivity"]) / len(scores["sensitivity"])
    scores["mean_specificity"] = sum(scores["specificity"]) / len(scores["specificity"])
    scores['f1_samples'] = f1_score(all_true, all_pred, average='samples', zero_division=0)
    scores['f1_macro'] = f1_score(all_true, all_pred, average='macro', zero_division=0)
    scores['f1_None'] = f1_score(all_true, all_pred, average=None, zero_division=0)
    scores['roc-auc-average'] = roc_auc_score(all_true, all_pred, multi_class='ovr')
    scores['roc-auc-elems'] = roc_auc_score(all_true, all_pred, average=None)

    return scores

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
            
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            true = y.cpu().numpy()
            
            all_preds.extend(preds)
            all_true.extend(true)
    
    return all_preds, all_true

if __name__ == "__main__":
    is_handcrafted = True
    _, test, _, names, features_num = get_dataloaders()
    if is_handcrafted:
        model = HandcraftedModel(in_channels=12, out_classes=len(names), handcrafted_classes=features_num)
        state_dict = torch.load("handcrafted_CNN_ECG_detection.pth", weights_only=False)
        model.load_state_dict(state_dict)
    else:
        model = BaseModel(in_channels=12, out_classes=len(names))
        state_dict = torch.load("CNN_ECG_detection.pth", weights_only=False)
        model.load_state_dict(state_dict)

    print("\n\n---------")
    print("SCORES:")
    print("---------\n")

    base_model_pred, all_pred = evaluate_model(model=model, test_loader=test, is_handcrafted=True)
    scores = basic_scores(all_pred, base_model_pred)

    for name in scores:
        print(f"{name}: {scores[name]}")

    
    


