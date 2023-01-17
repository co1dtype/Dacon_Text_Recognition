import torch.nn as nn

def get_loss():
    criterion = nn.CTCLoss(blank=0)
    return criterion
