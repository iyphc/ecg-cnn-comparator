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


def normalize_only_non_binary(feature_values):
    if feature_values.size == 0:
        return feature_values

    binary_features = []
    non_binary_features = []

    for i in range(feature_values.shape[1]):
        col = feature_values[:, i]
        unique_values = np.unique(col[~np.isnan(col)])
        if len(unique_values) <= 2 and all(val in [0, 1, -1] for val in unique_values):
            binary_features.append(col)
        else:
            non_binary_features.append(col)

    if binary_features:
        binary_features = np.column_stack(binary_features)
    else:
        binary_features = np.empty((feature_values.shape[0], 0))

    if non_binary_features:
        non_binary_features = np.column_stack(non_binary_features)
        non_binary_features = normalize(non_binary_features)
    else:
        non_binary_features = np.empty((feature_values.shape[0], 0))

    features = np.hstack([binary_features, non_binary_features])
    return features


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
    diseases=None,
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

    def aggregate_diagnostic(scp_dict, diseases):
        if not isinstance(scp_dict, dict):
            return []

        classes = set()
        for code, _ in scp_dict.items():
            if code not in agg_df.index:
                print(code)
                continue
            if diseases is not None:
                if code in diseases:
                    classes.add(code)
            else:
                classes.add(code)
        return list(classes) if classes else []

    Y["diagnostic_superclass"] = Y["scp_codes"].apply(
        lambda x: aggregate_diagnostic(x, diseases)
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

    handcrafted_train = normalize_only_non_binary(handcrafted_train)

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
    torch.save(mlb.classes_, "data/processed/diseases_names.pt")

    print("Data saved successfully!")
    return train_dataset, val_dataset, test_dataset, mlb.classes_, features


def load_ECG_dataset(
    path="data/raw/physionet.org/files/ptb-xl/1.0.1/",
    sampling_rate=100,
    diseases=None,
    features=None,
):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    diseases_names = None
    features_list = []
    if os.path.exists("data/processed/train_dataset.pt"):
        train_dataset = torch.load(
            "data/processed/train_dataset.pt", weights_only=False
        )
    if os.path.exists("data/processed/val_dataset.pt"):
        val_dataset = torch.load("data/processed/val_dataset.pt", weights_only=False)
    if os.path.exists("data/processed/test_dataset.pt"):
        test_dataset = torch.load("data/processed/test_dataset.pt", weights_only=False)
    if os.path.exists("data/processed/diseases_names.pt"):
        diseases_names = torch.load(
            "data/processed/diseases_names.pt", weights_only=False
        )
    if features:
        features_list = features

    if (
        (train_dataset is None)
        or (val_dataset is None)
        or (test_dataset is None)
        or (diseases_names is None)
        or not features_list
    ):
        print("THERE IS NO CORRECT DATASET")
        train_dataset, val_dataset, test_dataset, diseases_names, features_list = (
            process_dataset(
                path=path,
                sampling_rate=sampling_rate,
                diseases=diseases,
                features=features,
            )
        )

    print("Data loaded")
    return train_dataset, val_dataset, test_dataset, diseases_names, features_list


if __name__ == "__main__":
    process_dataset()
