import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

#DATASET_PATH = '../datasets/mvtec_anomaly_detection'
DATASET_PATH = '../datasets/multi_lighting'
class CustomMVTecDataset(Dataset):
    def __init__(self, dataset_path, class_name='bottle', is_train=True, resize=256, cropsize=256):
        if dataset_path is None:
            self.dataset_path = DATASET_PATH
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([
            T.Resize(resize, Image.ANTIALIAS),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([
            T.Resize(resize, Image.NEAREST),
            T.CenterCrop(cropsize),
            T.ToTensor()
        ])
        self.h = cropsize
        self.w = cropsize

    def __getitem__(self, idx):
        # Fetch the six image paths for the current index
        img_paths = self.x[idx]
        img_list = [Image.open(p).convert('RGB') for p in img_paths]
        imgs = torch.stack([self.transform_x(img) for img in img_list], dim=0)

        # Get the label and mask
        label = self.y[idx]
        mask_path = self.mask[idx]
        if label == 0:
            mask = torch.zeros([1, self.h, self.w])
        else:
            mask = Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask)

        return imgs, label, mask

    def __len__(self):
        return len(self.x)
        
    def get_size(self):

        return self.h, self.w

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([
                os.path.join(img_type_dir, f)
                for f in os.listdir(img_type_dir)
                if f.endswith('.png') and '_0' in f  # Fetching only the first image to get the base name
            ])
            for fpath in img_fpath_list:
                index = os.path.splitext(os.path.basename(fpath))[0].split('_')[0]
                x.append([os.path.join(img_type_dir, f"{index}_{i}.png") for i in range(6)])  # Six images
                label = 0 if img_type == 'good' else 1
                y.append(label)
                mask_path = os.path.join(gt_dir, img_type, f"{index}_mask.png") if label == 1 else None
                mask.append(mask_path)

        assert len(x) == len(y) and len(x) == len(mask), 'Mismatch in lengths of x, y, and mask lists'
        return x, y, mask
