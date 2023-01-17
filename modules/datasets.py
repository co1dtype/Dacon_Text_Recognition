import numpy as np
import cv2

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OCRDataset(Dataset):
    def __init__(self, df, cfg, txn, train_mode=True):
        self.df = df
        self.CFG = cfg
        self.txn = txn
        self.train_mode = train_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = Image.fromarray(
            cv2.imdecode(np.frombuffer(self.txn.get(f'image-{index + 1:09d}'.encode()), dtype=np.uint8),
                         cv2.IMREAD_COLOR))

        if self.train_mode:
            image = self.train_transform(img)
        else:
            image = self.test_transform(img)

        if self.train_mode:
            text = self.df[index]
            return image, text
        else:
            return image

    # Image Augmentation
    def train_transform(self, image):
        transform_ops = transforms.Compose([
            transforms.Resize((self.CFG.imgH, self.CFG.imgW)),
            transforms.ToTensor(),
            transforms.ColorJitter(),
            transforms.Normalize(mean=(0.8622502, 0.8622502, 0.8622502), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

    def test_transform(self, image):
        transform_ops = transforms.Compose([
            transforms.Resize((self.CFG.imgH, self.CFG.imgW)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.866558, 0.866558, 0.866558), std=(0.14509359, 0.14509359, 0.14509359))
        ])
        return transform_ops(image)