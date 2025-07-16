import os
import ast
import json
import wfdb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.constants import REDUCED_DISEASES_LIST

class ECG_Dataset(Dataset):
    def __init__(
        self,
        signals=None,
        handcrafted_features=None,
        labels=None,
        use_signals=True,
        use_handcrafted=True,
    ):
        self.use_signals = use_signals
        self.use_handcrafted = use_handcrafted

        if signals is not None:
            self.signals = torch.tensor(signals, dtype=torch.float32)
        else:
            self.signals = None

        if handcrafted_features is not None:
            self.handcrafted = torch.tensor(handcrafted_features, dtype=torch.float32)
        else:
            self.handcrafted = None

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            raise ValueError("Labels must be provided")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_signal = (
            self.signals[idx].transpose(0, 1)
            if self.use_signals and self.signals is not None
            else None
        )
        x_handcrafted = (
            self.handcrafted[idx] if self.use_handcrafted and self.handcrafted is not None else None
        )
        y = self.labels[idx]
        return x_signal, x_handcrafted, y

def visualise_ecg_by_id(df, ecg_id, sampling_rate, path):
    file = df.filename_lr.iloc[ecg_id] if sampling_rate == 100 else df.filename_hr.iloc[ecg_id]
    signal, fields = wfdb.rdsamp(path + file)
    wfdb.plot_items(
        signal=signal,
        fs=fields['fs'],
        sig_units=fields['units'],
        sig_name=fields['sig_name'],
        title='ECG Record',
        time_units='seconds',
        figsize=(10, 10),
        ecg_grids='all'
    )


def load_raw_data(df, sampling_rate, path):
    filenames = df.filename_lr if sampling_rate == 100 else df.filename_hr
    data = []
    
    for i, filename in enumerate(filenames):
        try:
            signal, _ = wfdb.rdsamp(path + filename)
            data.append(signal)
        except Exception as e:
            data.append(None)
            print(f"Failed to load file {filename}: {e}")
    
    if not data:
        raise ValueError("No data was successfully loaded")
    
    return np.array(data, dtype=np.float32)

def normalize(signals: np.ndarray) -> np.ndarray:
    mean = np.mean(signals, axis=1, keepdims=True)
    std = np.std(signals, axis=1, keepdims=True)

    # with np.errstate(divide='ignore', invalid='ignore'):
    #     X = np.where(std != 0, (signals - mean) / std, 0)

    X = np.where(std != 0, (signals - mean) / std, 0)
    return X

def handcrafted_extraction(df: pd.DataFrame, features):
    df = df.copy()
    
    # sex_mapping = {'male': 0, 'female': 1}
    # df['sex'] = df['sex'].map(sex_mapping).fillna(-1)
    # df['age'] = df['age'].fillna(df['age'].median())
    # df['height'] = df['height'].fillna(df['height'].median())
    # df['weight'] = df['weight'].fillna(df['weight'].median())
    
    # features = df[['age', 'sex', 'height', 'weight']].to_numpy(dtype=np.float32)

    features = df[features].to_numpy(dtype=np.float32)

    return binary_features, non_binary_features

def handle(path='data/raw/physionet.org/files/ptb-xl/1.0.1/',
            sampling_rate=100, reduced_dataset=None):
    
    print("STARTED PREPAIRING DATASET\n")
    if not os.path.exists(path + 'ptbxl_database.csv'):
        raise FileNotFoundError(f"Database file not found at {path}ptbxl_database.csv")
    
    if not os.path.exists(path + 'scp_statements.csv'):
        raise FileNotFoundError(f"SCP statements file not found at {path}scp_statements.csv")

    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y['scp_codes'] = Y['scp_codes'].apply(ast.literal_eval)
    
    X = load_raw_data(Y, sampling_rate, path)
    X_binary, X_non_binary = handcrafted_extraction(Y)

    X_non_binary = normalize(X_non_binary)
    X = normalize(X)
    X_handcrafted = np.hstack([X_binary, X_non_binary])
    
    features_num = X_handcrafted.shape[1]

    # Очистка данных от None значений
    idx = [i for i, x in enumerate(X) if x is not None]
    X = X[idx]
    X_handcrafted = X_handcrafted[idx]
    Y = Y.iloc[idx]


    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)

    def aggregate_diagnostic(scp_dict, reduced_dataset):
        if not isinstance(scp_dict, dict):
            return []
        
        classes = set()
        for code, _ in scp_dict.items():
            if code not in agg_df.index:
                print(code)
                continue
            if reduced_dataset is not None:
                if code in reduced_dataset:
                    classes.add(code)
            else:
                classes.add(code)
        return list(classes) if classes else []

    Y['diagnostic_superclass'] = Y['scp_codes'].apply(lambda x: aggregate_diagnostic(x, reduced_dataset))

    mlb = MultiLabelBinarizer()
    y_onehot = mlb.fit_transform(Y['diagnostic_superclass'])
    y_onehot = np.array(y_onehot)
    Y['one_hot'] = [row.tolist() for row in y_onehot]

    test_fold = 10
    mask_train = Y['strat_fold'] != test_fold
    mask_test = Y['strat_fold'] == test_fold

    X_train = X[mask_train.to_numpy()]
    X_test = X[mask_test.to_numpy()]

    y_train = np.array(Y[mask_train]['one_hot'].tolist(), dtype=np.float32)
    y_test = np.array(Y[mask_test]['one_hot'].tolist(), dtype=np.float32)

    os.makedirs("data/processed", exist_ok=True)

    train_df = pd.DataFrame(y_train, columns=mlb.classes_)
    test_df = pd.DataFrame(y_test, columns=mlb.classes_)
    
    handcrafted_train = X_handcrafted[mask_train.to_numpy()]
    handcrafted_test = X_handcrafted[mask_test.to_numpy()]

    train_dataset = ECG_Dataset(signals=X_train, labels=y_train, handcrafted_features=handcrafted_train)
    test_dataset = ECG_Dataset(signals=X_test, labels=y_test, handcrafted_features=handcrafted_test)

    if (not os.path.exists('data/processed/train_dataset.pt')):
        torch.save(train_dataset, 'data/processed/train_dataset.pt')
    if (not os.path.exists('data/processed/test_dataset.pt')):
        torch.save(test_dataset, 'data/processed/test_dataset.pt')
    if (not os.path.exists('data/processed/diseases_names.pt')):
        torch.save(mlb.classes_, 'data/processed/diseases_names.pt')
    if (not os.path.exists('data/processed/features_num.pt')):
        with open("data/processed/features_num.json", "w") as f:
            json.dump({"features_num": features_num}, f, indent=4)
    
    print("Data saved successfully!")
    return train_dataset, test_dataset, mlb.classes_, features_num

def load_ECG_dataset(path='data/raw/physionet.org/files/ptb-xl/1.0.1/',
            sampling_rate=100, reduced_dataset=None):
    train_dataset = None
    test_dataset = None
    diseases_names = None
    features_num = -1
    if (os.path.exists('data/processed/train_dataset.pt')):
        train_dataset = torch.load('data/processed/train_dataset.pt', weights_only=False)
    if (os.path.exists('data/processed/test_dataset.pt')):
        test_dataset = torch.load('data/processed/test_dataset.pt', weights_only=False)
    if (os.path.exists('data/processed/diseases_names.pt')):
        diseases_names = torch.load('data/processed/diseases_names.pt', weights_only=False)
    if (os.path.exists('data/processed/features_num.json')):
        with open('data/processed/features_num.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            features_num = data["features_num"]
        
    if ((train_dataset is None) or (test_dataset is None) or (diseases_names is None) or features_num == -1):
        print("THERE IS NO CORRECT DATASET")
        train_dataset, test_dataset, diseases_names, features_num = handle(path=path, sampling_rate=sampling_rate, reduced_dataset=reduced_dataset)
    
    print("Data loaded")
    return train_dataset, test_dataset, diseases_names, features_num
    

if __name__ == "__main__":
    import time
    start = time.time()
    load_ECG_dataset()
    end = time.time()
    print(round(end-start, 2))