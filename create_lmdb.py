import os, sys, argparse, lmdb, fire
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

def get_train_val_df(path, seed=42):
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
        img_path = f"./data/{img_path[2:]}"
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


def create_and_write_lmdb(path, data_type, df):
    env = lmdb.open(path, map_size=1099511627776)
    write_lmdb(df, env, bar_desc=data_type)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', type=str, default='', help='input save directory')
    parser.add_argument('--seed', type=int, default=42, help='input seed')
    parser.add_argument('--test', type=bool, default=False, help='input train and test boolean')

    
    CFG.dirs = f"./{CFG.dirs}"

    if not CFG.test:
        train_dir = os.path.join(CFG.dirs, 'data', 'lmdb', 'train_lmdb')
        val_dir = os.path.join(CFG.dirs, 'data', 'lmdb', 'val_lmdb')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        train, val = get_train_val_df(CFG.dirs, CFG.seed)
        print(train_dir)
        fire.Fire(create_and_write_lmdb(train_dir, "train", train), command='')
        fire.Fire(create_and_write_lmdb(val_dir, "val", val), command='')
    else:
        test_dir = os.path.join(CFG.dirs, 'data', 'lmdb', 'val_lmdb')
        os.makedirs(test_dir, exist_ok=True)

        test_path = CFG.dirs = os.path.join(prj_dir, 'data', 'test.csv')
        test = pd.read_csv(test_path)
        fire.Fire(create_and_write_lmdb(test_dir, "test", test), command='')
