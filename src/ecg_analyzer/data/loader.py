import torch
from .preprocess import load_ECG_dataset
from torch.utils.data import DataLoader, random_split


def get_dataloaders(
    batch_size=128,
    num_workers=2,
    raw_path="data/raw/physionet.org/files/ptb-xl/1.0.1/",
    sampling_rate=100,
    reduced_dataset=None,
    features=None,
):
    train_set, val_set, test_set, classes, features_list = load_ECG_dataset(
        path=raw_path,
        sampling_rate=sampling_rate,
        reduced_dataset=reduced_dataset,
        features=features,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, val_loader, classes, features_list
