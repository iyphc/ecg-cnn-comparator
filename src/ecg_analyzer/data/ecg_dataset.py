import torch
from torch.utils.data import Dataset
import wfdb


class ECG_Dataset(Dataset):
    def __init__(
        self,
        signals_names=None,
        handcrafted_features=None,
        labels=None,
        use_signals=True,
        use_handcrafted=True,
        is_train=False,
    ):
        self.use_signals = use_signals
        self.use_handcrafted = use_handcrafted
        self.filenames = signals_names
        self.is_train = is_train

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
        signal, _ = wfdb.rdsamp(self.filenames[idx])
        signal = torch.from_numpy(signal).float()
        signal = signal.transpose(0, 1)

        if self.is_train:
            mean = signal.mean(dim=1, keepdim=True)
            std = signal.std(dim=1, keepdim=True)
            eps = 1e-8
            signal = (signal - mean) / (std + eps)

        x_handcrafted = self.handcrafted[idx] if self.handcrafted is not None else None
        y = self.labels[idx]
        return signal, x_handcrafted, y
