import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def save_model(model, path):
    torch.save({
            'model_state_dict': model.state_dict(), 
            'threshold': model.threshold}, path
        )
def load_model(model, path):
    state_dict = torch.load(path, weights_only=False)
    model.load_state_dict(state_dict['model_state_dict'])
    model.threshold = state_dict['threshold']
