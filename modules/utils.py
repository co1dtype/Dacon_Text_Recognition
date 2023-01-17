import os, random, torch, lmdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

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


def get_train_val(path, seed=42):
    train_path = os.path.join(path, 'data', 'train.csv')
    df = pd.read_csv(train_path)

    df['len'] = df['label'].str.len()
    train_v1 = df[df['len'] == 1]

    df = df[df['len'] > 1]
    train_v2, val, _, _ = train_test_split(df, df['len'], test_size=0.1295, random_state=seed)
    train = pd.concat([train_v1, train_v2])

    return (train, val)

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def write_lmdb(df, env, bar_desc='train'):
    cnt = 1
    cache = {}
    for i in tqdm(range(len(df)), desc=bar_desc):
        img_path = df.iloc[i]['img_path']
        if bar_desc != "test":
            label = df.iloc[i]['label']

        with open(img_path, 'rb') as f:
            img_bin = f.read()

        image_key = f'image-{cnt:09d}'.encode()
        label_key = f'label-{cnt:09d}'.encode()

        cache[image_key] = img_bin
        if bar_desc != "test":
            cache[label_key] = label.encode()

        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
        cnt += 1

    # remain cache
    cache['num-samples'.encode()] = str(cnt - 1).encode()
    write_cache(env, cache)

def create_lmdb(path, data_type, df):
    data_path = os.path.join(path, 'data', 'lmdb', f'{data_type}_lmdb')
    env = lmdb.open(data_path, map_size=1099511627776)
    write_lmdb(df, env, bar_desc=data_type)
    env.close()



