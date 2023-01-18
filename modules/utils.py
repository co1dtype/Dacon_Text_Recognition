import os, random, torch, create_lmdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

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


def submit_csv(submit, predictions):
    submit['label'] = predictions
    submit['label'] = submit['label'].apply(correct_prediction)
    submit.to_csv('./submission.csv', index=False)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_val_df(path, seed=42):
    train_path = os.path.join(path, 'data', 'train.csv')
    df = pd.read_csv(train_path)

    df['len'] = df['label'].str.len()
    train_v1 = df[df['len'] == 1]

    df = df[df['len'] > 1]
    train_v2, val, _, _ = train_test_split(df, df['len'], test_size=0.1295, random_state=seed)
    train = pd.concat([train_v1, train_v2])

    return (train, val)


def get_train_val_df(path, seed=42):
    train_path = os.path.join(path, 'data', 'train.csv')
    df = pd.read_csv(train_path)

    df['len'] = df['label'].str.len()
    train_v1 = df[df['len'] == 1]
    
    df = df[df['len'] > 1]
    train_v2, val, _, _ = train_test_split(df, df['len'], test_size=0.1295, random_state=seed)
    train = pd.concat([train_v1, train_v2])

    return (train, val)
