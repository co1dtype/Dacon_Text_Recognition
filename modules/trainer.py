import torch
import numpy as np
from tqdm.auto import tqdm
from metrics import compute_loss, compute_acc
from recoder import Recode


def train(model, epochs, scaler, idx2char, criterion, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)

    best_model = None
    early_stop = 5
    best_acc = -1
    best_loss = 1e9

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = []
        train_acc = []
        model_save = False
        for image_batch, text_batch in tqdm(iter(train_loader)):
            image_batch = image_batch.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                text_batch_logits = model(image_batch)
                loss = compute_loss(text_batch, text_batch_logits, criterion, device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_acc += compute_acc(text_batch, text_batch_logits, idx2char)
            train_loss.append(loss.item())

        _train_loss = np.mean(train_loss)

        _train_acc = 100. * sum(train_acc) / len(train_loader.dataset)
        _val_loss, _val_acc = validation(model, val_loader, criterion, idx2char, device)

        print(f'Epoch : [{epoch}] Train CTC Loss : [{_train_loss:.5f}] Val CTC Loss : [{_val_loss:.5f}]')
        print(f'Epoch : [{epoch}] Train Accuracy : [{_train_acc:.5f}] Val Accuracy : [{_val_acc:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_loss)

        if best_loss > _val_loss or best_acc < _val_acc:
            early_stop += 1

        if best_loss > _val_loss:
            best_loss = _val_loss

        if best_acc < _val_acc:
            best_acc = _val_acc
            best_model = model
            model_save = True

        early_stop -= 1
        Recode(epoch, model, _train_loss, _val_loss, _train_acc, _val_acc, best_loss, best_acc, model_save)

        if early_stop == 0:
            break

    return best_model


def validation(model, val_loader, criterion, idx2char, device):
    model.eval()
    val_loss = []
    val_acc = []
    with torch.no_grad():
        for image_batch, text_batch in tqdm(iter(val_loader)):
            image_batch = image_batch.to(device)

            with torch.cuda.amp.autocast():
                text_batch_logits = model(image_batch)
                loss = compute_loss(text_batch, text_batch_logits, criterion, device)

            val_loss.append(loss.item())
            val_acc += compute_acc(text_batch, text_batch_logits, idx2char)

    _val_loss = np.mean(val_loss)
    _val_acc = 100. * sum(val_acc) / len(val_loader.dataset)
    return _val_loss, _val_acc