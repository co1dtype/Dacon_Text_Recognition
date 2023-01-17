import os
import matplotlib as plt
import pandas as pd
import torch

from datetime import datetime

def Recode(epoch, model, CFG, train_loss, val_loss, train_acc, val_acc, best_loss, best_acc, model_save=False):
    current_epoch = 1
    recode_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss',
                                      'train_acc', 'val_acc', 'best_loss',
                                      'best_acc'])
    dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(1, 2)
    plt.close(fig)

    # Data Write
    os.makedirs(f"./result/{dir_name}", exist_ok=True)
    new_data = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "best_loss": best_loss,
        "best_acc": best_acc
    }

    recode_df = recode_df.append(new_data, ignore_index=True)
    recode_df.to_csv(f'./result/{dir_name}/recode.csv', index=False)

    # Data Visualization
    os.makedirs(f"./result/{dir_name}/plot", exist_ok=True)
    fig, axes = plt.subplots(1, 2)
    fig = plt.figure(figsize=(20, 7))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(recode_df['train_loss'].to_list(), label="train loss")
    ax1.plot(recode_df['val_loss'].to_list(), label="val loss")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(recode_df['train_acc'], label="train Accuracy")
    ax2.plot(recode_df['val_acc'], label="val Accuracy")
    ax2.legend()

    if os.path.isfile(f"./result/{dir_name}/plot/loss_and_acc.png"):
        os.remove(f"./result/{dir_name}/plot/loss_and_acc.png")
    plt.savefig(f"./result/{dir_name}/plot/loss_and_acc.png")

    # Write Hyperparameter
    if epoch == 1:
        with open(f"./result/{dir_name}/train.txt", 'w', encoding='UTF-8') as f:
            for key, values in CFG.items():
                f.write(f"{key} {values}" + "\n")

    if model_save:
        torch.save(model, f"./result/{dir_name}/model.pt")

