from torch.optim import lr_scheduler


def get_scheduler(scheduler_str) -> object:
    scheduler = None

    if scheduler_str == 'CosineAnnealingLR':

        scheduler = lr_scheduler.CosineAnnealingLR

    elif scheduler_str == 'CosineAnnealingWarmRestarts':

        scheduler = lr_scheduler.CosineAnnealingWarmRestarts

    elif scheduler_str == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau

    if scheduler is None:
        raise Exception(f"'{scheduler}': This optimizer does not exist.")
    return scheduler