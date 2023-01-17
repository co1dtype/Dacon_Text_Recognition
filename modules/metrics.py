import torch
import torch.nn.functional as F

def encode_text_batch(text_batch, char2idx):
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)

    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)

    return text_batch_targets, text_batch_targets_lens


def compute_loss(text_batch, text_batch_logits, criterion, device):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                       fill_value=text_batch_logps.size(0),
                                       dtype=torch.int32).to(device) # [batch_size]

    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss



def correct_prediction(word):
    parts = word.split("-")

    def remove_duplicates(text):
        if len(text) > 1:
            letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx - 1]]
        elif len(text) == 1:
            letters = [text[0]]
        else:
            return ""
        return "".join(letters)
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word


def compute_acc(text_batch, text_batch_logits, idx2char):
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2)  # [T, batch_size]
    text_batch_tokens = text_batch_tokens.T  # [batch_size, T]

    check = [];
    cnt = 0
    for text_tokens, answer in zip(text_batch_tokens, text_batch):
        text = [idx2char[int(idx)] for idx in text_tokens]
        text = "".join(text)

        check.append(1 if correct_prediction(text) == answer else 0)

    return check
