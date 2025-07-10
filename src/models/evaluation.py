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

def confusion_matrix(all_true, all_pred):
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    matrix = {}
    matrix["TP"] = np.sum((all_true == 1) & (all_pred == 1), axis = 0)
    matrix["FP"] = np.sum((all_true == 0) & (all_pred == 1), axis = 0)
    matrix["FN"] = np.sum((all_true == 1) & (all_pred == 0), axis = 0)
    matrix["TN"] = np.sum((all_true == 0) & (all_pred == 0), axis = 0)

    return matrix


def basic_scores(all_true, all_pred):
    scores = {}

    matrix = confusion_matrix(all_true, all_pred)
    eps = 1e-8
    
    scores["sensitivity"] = matrix["TP"] / (matrix["TP"] + matrix["FN"] + eps)
    scores["specificity"] = matrix["TN"] / (matrix["TN"] + matrix["FP"] + eps)
    scores["mean_sensitivity"] = sum(scores["sensitivity"]) / len(scores["sensitivity"])
    scores["mean_specificity"] = sum(scores["specificity"]) / len(scores["specificity"])
    scores['f1_samples'] = f1_score(all_true, all_pred, average='samples')
    scores['f1_macro'] = f1_score(all_true, all_pred, average='macro')
    scores['f1_None'] = f1_score(all_true, all_pred, average=None)
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
    
    scores = basic_scores(all_true, all_preds)
    return scores

if __name__ == "__main__":
    is_handcrafted = True
    _, test, _, names = get_dataloaders()
    if is_handcrafted:
        handcrafted_size = next(iter(test))[1].shape[1]
        model = HandcraftedModel(in_channels=12, out_classes=len(names), handcrafted_classes=handcrafted_size)
        state_dict = torch.load("handcrafted_CNN_ECG_detection.pth", weights_only=False)
        model.load_state_dict(state_dict)
    else:
        model = BaseModel(in_channels=12, out_classes=len(names))
        state_dict = torch.load("CNN_ECG_detection.pth", weights_only=False)
        model.load_state_dict(state_dict)

    scores = evaluate_model(model=model, test_loader=test, is_handcrafted=True)
    print("\n\n---------")
    print("SCORES:")
    print("---------\n")
    print(f"Sensitivity: {scores["sensitivity"]}")
    print(f"Specificity: {scores["specificity"]}\n")
    print(f"Mean sensitivity: {scores["mean_sensitivity"]:.2%}")
    print(f"Mean specificity: {scores["mean_specificity"]:.2%}\n")
    print(f"F1: {scores["f1_None"]}")
    print(f"F1_samples: {scores["f1_samples"]:.2%}")
    print(f"F1_macro: {scores["f1_macro"]:.2%}\n")
    print(f"ROC-AUC: {scores["roc-auc-average"]:.2}")
    print(f"ROC-AUC-elem: {scores["roc-auc-elems"]}")


