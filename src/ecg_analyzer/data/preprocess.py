import os
import ast
import json
import wfdb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from .ecg_dataset import ECG_Dataset


def visualise_ecg_by_id(df, ecg_id, sampling_rate, path):
    file = (
        df.filename_lr.iloc[ecg_id]
        if sampling_rate == 100
        else df.filename_hr.iloc[ecg_id]
    )
    signal, fields = wfdb.rdsamp(path + file)
    wfdb.plot_items(
        signal=signal,
        fs=fields["fs"],
        sig_units=fields["units"],
        sig_name=fields["sig_name"],
        title="ECG Record",
        time_units="seconds",
        figsize=(10, 10),
        ecg_grids="all",
    )


def load_data_names(df, sampling_rate, path):
    filenames = df.filename_lr if sampling_rate == 100 else df.filename_hr

    data = []

    for filename in filenames:
        data.append(path + filename)

    return data


def normalize(signals: np.ndarray) -> np.ndarray:
    mean = np.mean(signals, axis=1, keepdims=True)
    std = np.std(signals, axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.where(std != 0, (signals - mean) / std, 0)

    return X


def split_and_fill(features):
    isnan = np.isnan(features)
    unique_values = [np.unique(col[~isnan[:, j]]) for j, col in enumerate(features.T)]
    is_binary = np.array([len(u) <= 2 and set(u) <= {-1, 0, 1} for u in unique_values])

    for j in np.where(is_binary)[0]:
        col = features[:, j]
        nan_idx = isnan[:, j]
        if nan_idx.all():
            fill = 0.0
        else:
            vals, counts = np.unique(col[~nan_idx], return_counts=True)
            fill = vals[np.argmax(counts)]
        col[nan_idx] = fill
    for j in np.where(~is_binary)[0]:
        col = features[:, j]
        nan_idx = isnan[:, j]
        col[nan_idx] = np.nanmean(col)

    return features


def normalize_non_binary(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X

    split_and_fill(X)

    unique_values = [np.unique(col) for j, col in enumerate(X.T)]
    is_binary = np.array([len(u) <= 2 and set(u) <= {-1, 0, 1} for u in unique_values])

    nb_cols = np.where(~is_binary)[0]
    if nb_cols.size:
        X[:, nb_cols] = normalize(X[:, nb_cols])

    return X


def handcrafted_extraction(df: pd.DataFrame, features):
    df = df.copy()

    feature_values = []

    for feature in features:
        if feature not in df.columns:
            print(f"Признак '{feature}' не найден в датафрейме")
            continue
        feature_values.append(df[feature].values)

    return feature_values


def process_dataset(
    path="data/raw/physionet.org/files/ptb-xl/1.0.1/",
    sampling_rate=100,
    pathologies=None,
    features=None,
):

    print("STARTED PREPAIRING DATASET\n")
    if not os.path.exists(path + "ptbxl_database.csv"):
        raise FileNotFoundError(f"Database file not found at {path}ptbxl_database.csv")

    if not os.path.exists(path + "scp_statements.csv"):
        raise FileNotFoundError(
            f"SCP statements file not found at {path}scp_statements.csv"
        )

    features = features if features else []

    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y["scp_codes"] = Y["scp_codes"].apply(ast.literal_eval)

    X = load_data_names(Y, sampling_rate, path)
    X = np.array(X)

    X_handcrafted = handcrafted_extraction(Y, features)
    X_handcrafted = np.array(X_handcrafted)

    if X_handcrafted.size > 0:
        X_handcrafted = X_handcrafted.T
    else:
        X_handcrafted = np.zeros((len(X), 0))

    idx = [i for i, x in enumerate(X) if x is not None]
    X = X[idx]
    X_handcrafted = X_handcrafted[idx]
    Y = Y.iloc[idx]

    agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)

    def aggregate_diagnostic(scp_dict, pathologies):
        if not isinstance(scp_dict, dict):
            return []

        classes = set()
        for code, _ in scp_dict.items():
            if code not in agg_df.index:
                print(code)
                continue
            if pathologies is not None:
                if code in pathologies:
                    classes.add(code)
            else:
                classes.add(code)
        return list(classes) if classes else []

    Y["diagnostic_superclass"] = Y["scp_codes"].apply(
        lambda x: aggregate_diagnostic(x, pathologies)
    )

    mlb = MultiLabelBinarizer()
    y_onehot = mlb.fit_transform(Y["diagnostic_superclass"])
    y_onehot = np.array(y_onehot)
    Y["one_hot"] = [row.tolist() for row in y_onehot]

    test_fold = 10
    val_fold = 9
    mask_train = (Y["strat_fold"] != test_fold) & (Y["strat_fold"] != val_fold)
    mask_val = Y["strat_fold"] == val_fold
    mask_test = Y["strat_fold"] == test_fold

    X_train = X[mask_train.to_numpy()]
    X_val = X[mask_val.to_numpy()]
    X_test = X[mask_test.to_numpy()]

    handcrafted_train = X_handcrafted[mask_train.to_numpy()]
    handcrafted_val = X_handcrafted[mask_val.to_numpy()]
    handcrafted_test = X_handcrafted[mask_test.to_numpy()]

    handcrafted_train = normalize_non_binary(handcrafted_train)
    handcrafted_val = normalize_non_binary(handcrafted_val)
    handcrafted_test = normalize_non_binary(handcrafted_test)

    y_train = np.array(Y[mask_train]["one_hot"].tolist(), dtype=np.float32)
    y_val = np.array(Y[mask_val]["one_hot"].tolist(), dtype=np.float32)
    y_test = np.array(Y[mask_test]["one_hot"].tolist(), dtype=np.float32)

    os.makedirs("data/processed", exist_ok=True)

    train_dataset = ECG_Dataset(
        signals_names=X_train, labels=y_train, handcrafted_features=handcrafted_train
    )
    val_dataset = ECG_Dataset(
        signals_names=X_val, labels=y_val, handcrafted_features=handcrafted_val
    )
    test_dataset = ECG_Dataset(
        signals_names=X_test, labels=y_test, handcrafted_features=handcrafted_test
    )

    # Всегда сохраняем файлы (перезаписываем существующие)
    torch.save(train_dataset, "data/processed/train_dataset.pt")
    torch.save(val_dataset, "data/processed/val_dataset.pt")
    torch.save(test_dataset, "data/processed/test_dataset.pt")
    torch.save(mlb.classes_, "data/processed/pathologies_names.pt")

    print("Data saved successfully!")
    return train_dataset, val_dataset, test_dataset, mlb.classes_, features


def load_ECG_dataset(
    path="data/raw/physionet.org/files/ptb-xl/1.0.1/",
    sampling_rate=100,
    pathologies=None,
    features=None,
):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    pathologies_names = None
    features_list = []
    if os.path.exists("data/processed/train_dataset.pt"):
        train_dataset = torch.load(
            "data/processed/train_dataset.pt", weights_only=False
        )
    if os.path.exists("data/processed/val_dataset.pt"):
        val_dataset = torch.load("data/processed/val_dataset.pt", weights_only=False)
    if os.path.exists("data/processed/test_dataset.pt"):
        test_dataset = torch.load("data/processed/test_dataset.pt", weights_only=False)
    if os.path.exists("data/processed/pathologies_names.pt"):
        pathologies_names = torch.load(
            "data/processed/pathologies_names.pt", weights_only=False
        )
    if features:
        features_list = features

    if (
        (train_dataset is None)
        or (val_dataset is None)
        or (test_dataset is None)
        or (pathologies_names is None)
        or not features_list
    ):
        print("THERE IS NO CORRECT DATASET")
        train_dataset, val_dataset, test_dataset, pathologies_names, features_list = (
            process_dataset(
                path=path,
                sampling_rate=sampling_rate,
                pathologies=pathologies,
                features=features,
            )
        )

    print("Data loaded")
    return train_dataset, val_dataset, test_dataset, pathologies_names, features_list
