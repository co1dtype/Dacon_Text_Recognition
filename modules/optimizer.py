import torch.optim as optim

def get_optimizer(optimizer_str: str) -> 'optimizer':
    optimizer = None
    if optimizer_str == 'Adadelta':
        optimizer = optim.Adadelta

    elif optimizer_str == 'Adam':
        optimizer = optim.Adam

    elif optimizer_str == 'AdamW':
        optimizer = optim.AdamW

    if optimizer is None:
        raise Exception(f"'{optimizer_str}': This optimizer does not exist.")
    return optimizer
