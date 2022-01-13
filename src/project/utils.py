import torch


def save_model(model, path):
    torch.save({'model_state_dict': model.state_dict()}, path)


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
