import torch, os, sys, argparse, lmdb
from modules.utils import get_train_val_df
from modules.datasets import OCRDataset
from modules.losses import get_loss
from modules.optimizer import get_optimizer
from modules.schedulers import get_scheduler
from modules.trainer import Trainer
from models.utils import get_model
from torch.utils.data import DataLoader
from torchsummary import summary
from datetime import datetime

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_worker', type=str, help='input num worker')
    parser.add_argument('--opti', type=str, default='AdamW', help='input optimizer')
    parser.add_argument('--lr', type=float, default=3e-3, help='input learning rate')
    parser.add_argument('--seed', type=int, default=42, help='input seed')
    parser.add_argument('--epoch', type=int, default=20, help='input epoch')
    """ Data processing """
    parser.add_argument('--imgH', type=int, default=128, help='the height of the input imgH')
    parser.add_argument('--imgW', type=int, default=332, help='the width of the input imgW')
    """ Model Architecture """
    parser.add_argument('--model', type=str, help='input model name')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    CFG = parser.parse_args()
    dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = CFG.num_worker
    
    # Data split
    train, val = get_train_val_df(prj_dir, CFG.seed)

    # label preprocessing
    train_gt = [gt for gt in train['label']]
    train_gt = "".join(train_gt)
    letters = sorted(list(set(list(train_gt))))

    vocabulary = ["-"] + letters
    idx2char = {k: v for k, v in enumerate(vocabulary, start=0)}
    char2idx = {v: k for k, v in idx2char.items()}

    # Data Load
    train_path = os.path.join(prj_dir, 'data', 'lmdb', f'train_lmdb')
    env = lmdb.open(train_path, readonly=True, lock=False)
    txn = env.begin()
    train_dataset = OCRDataset(train['label'].values, CFG, txn)

    val_path = os.path.join(prj_dir, 'data', 'lmdb', f'val_lmdb')
    env = lmdb.open(val_path, readonly=True, lock=False)
    txn = env.begin()
    val_dataset = OCRDataset(val['label'].values, CFG, txn)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=True)

    # Hyperparameter
    model = get_model(CFG.model)
    model = model(len(char2idx), CFG.hidden_size).eval()
    criterion = get_loss()
    scaler = torch.cuda.amp.GradScaler()

    optimizer = get_optimizer(CFG.opti)
    optimizer = optimizer(params = model.parameters(), lr = CFG.lr, weight_decay=0.3)

    scheduler = get_scheduler('ReduceLROnPlateau')
    scheduler = scheduler(optimizer, mode='min', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

    print(summary(model, input_size=(3, CFG.imgH, CFG.imgW), batch_size=CFG.batch_size,
                  device="cpu"))


    # Run
    Trainer(model, CFG.epoch, dir_name, CFG, scaler, char2idx, idx2char, criterion, optimizer, train_loader, val_loader, scheduler, device)



