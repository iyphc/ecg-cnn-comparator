from ..data.loader import get_dataloaders
from ..utils.utils import get_device
from ..models.base_model import BaseModel
from ..models.cnn_handcrafted import HandcraftedModel
import tqdm
import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix
from collections import defaultdict


def confusion_matrix(all_true, all_pred):
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    matrix = {}
    matrix["TP"] = np.sum((all_true == 1) & (all_pred == 1), axis=0)
    matrix["FP"] = np.sum((all_true == 0) & (all_pred == 1), axis=0)
    matrix["FN"] = np.sum((all_true == 1) & (all_pred == 0), axis=0)
    matrix["TN"] = np.sum((all_true == 0) & (all_pred == 0), axis=0)
    return matrix


def specificity_score(y_true, y_pred, average: str = "macro"):

    matrix = confusion_matrix(y_true, y_pred)

    specificity = None

    if average == "macro":
        tmp = matrix["TN"] / (matrix["TN"] + matrix["FP"])
        specificity = tmp.sum() / tmp.shape[0]
    elif average == "micro":
        TN_sum = matrix["TN"].sum()
        FP_sum = matrix["FP"].sum()
        specificity = TN_sum / (TN_sum + FP_sum)
    elif average == "weighted":
        weights = matrix["TN"] + matrix["FP"]
        specificity_none = np.where(weights == 0, 0.0, matrix["TN"] / weights)
        if weights.sum() == 0:
            specificity = 0.0
        else:
            specificity = np.average(specificity_none, weights=weights)
    elif average == None:
        specificity = matrix["TN"] / (matrix["TN"] + matrix["FP"])
    else:
        raise ValueError(f"Unknown average type: {average}")

    return specificity


def basic_scores(all_true, all_pred, threshold=0.5):
    scores = {}

    all_pred_prob = np.array(all_pred)
    all_pred = np.array(all_pred) > threshold

    scores["recall_micro"] = recall_score(
        all_true, all_pred, average="micro", zero_division=0
    )
    scores["recall_macro"] = recall_score(
        all_true, all_pred, average="macro", zero_division=0
    )
    scores["recall_weighted"] = recall_score(
        all_true, all_pred, average="weighted", zero_division=0
    )
    scores["recall_none"] = recall_score(
        all_true, all_pred, average=None, zero_division=0
    )
    scores["specificity_micro"] = specificity_score(all_true, all_pred, average="micro")
    scores["specificity_macro"] = specificity_score(all_true, all_pred, average="macro")
    scores["specificity_weighted"] = specificity_score(
        all_true, all_pred, average="weighted"
    )
    scores["specificity_none"] = specificity_score(all_true, all_pred, average=None)
    scores["f1_micro"] = f1_score(all_true, all_pred, average="micro", zero_division=0)
    scores["f1_macro"] = f1_score(all_true, all_pred, average="macro", zero_division=0)
    scores["f1_weighted"] = f1_score(
        all_true, all_pred, average="weighted", zero_division=0
    )
    scores["f1_none"] = f1_score(all_true, all_pred, average=None, zero_division=0)
    scores["roc-auc-macro"] = roc_auc_score(
        all_true, all_pred_prob, multi_class="ovo", average="macro"
    )
    scores["roc-auc-elems"] = roc_auc_score(all_true, all_pred_prob, average=None)

    return scores


def compare_models(
    base_model, handcrafted_model, test_loader, seed=52, repet_number=1000, alpha=0.05
):
    np.random.seed(seed)
    base_model_pred, all_true = evaluate_model(base_model, test_loader)
    handcrafted_model_pred, _ = evaluate_model(
        handcrafted_model, test_loader, is_handcrafted=True
    )
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
            if not is_handcrafted:
                outputs = model(X)
            else:
                X_handcrafted = X_handcrafted.to(device)
                outputs = model(X, X_handcrafted)

            preds = torch.sigmoid(outputs).cpu().numpy()
            true = y.cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(true)

    return all_preds, all_true
