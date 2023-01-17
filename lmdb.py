import os, sys, argparse, fire
import pandas as pd
from modules.utils import get_train_val, create_lmdb

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', type=str, default='', help='input save directory')
    parser.add_argument('--seed', type=int, default=42, help='input seed')
    parser.add_argument('--test', type=bool, default=False, help='input train and test boolean')

    CFG = parser.parse_args()

    if not CFG.dirs:
        CFG.dirs = os.path.join(prj_dir, *CFG.dirs.split('/'))
    else:
        CFG.dirs = prj_dir

    if CFG.train:
        train, val = get_train_val(CFG.dirs, CFG.seed)
        fire.Fire(create_lmdb(CFG.dirs, "train", train), command='')
        fire.Fire(create_lmdb(CFG.dirs, "train", val), command='')
    else:
        test_path = CFG.dirs = os.path.join(prj_dir, 'data', 'test.csv')
        test = pd.read_csv(test_path)
        fire.Fire(create_lmdb(CFG.dirs, "test", test), command='')
