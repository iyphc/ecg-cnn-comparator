import torch.nn as nn

class HandcraftedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tmp = 0