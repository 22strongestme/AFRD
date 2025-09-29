import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import platform

ret = {}
# 当前操作系统
plat_form = platform.platform()
if "Linux" in plat_form:
    ret["plat_form"] = "Linux"
elif "Windows" in plat_form:
    ret["plat_form"] = "Windows"
else:
    ret['plat_form'] = "Mac"


CLASS_NAMES = ['cover_patch', 'cover_whole']

DATASET_PATH = '../datasets/covercap_1010'
# if ret["plat_form"] == 'Linux':
#     dataset_path = '../datasets/covercap_1010'
# else:
#     dataset_path = 'G:/images/covercap'

class CoverCapDataset(Dataset):
    def __init__(self, dataset_path=None, class_name='cover_patch', is_train=True,
                 resize=256, cropsize=256, do_resize=True):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)

        if dataset_path is None:
            self.dataset_path = DATASET_PATH
        else:
            self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x, self.y, self.mask, self.name = self.load_dataset_folder()

        # set transforms
        if do_resize:
            # set transforms
            self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                          T.CenterCrop(cropsize),
                                          T.ToTensor(),
                                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])
            self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                             T.CenterCrop(cropsize),
                                             T.ToTensor()])
            self.h = cropsize
            self.w = cropsize
        else:
            # calculate size
            x = self.x[0]
            x = Image.open(x).convert('RGB')
            self.w, self.h = x.size

            # should be times of 32
            # self.h = self.h // 32 * 32
            # self.w = self.w // 32 * 32

            self.transform_x = T.Compose([T.CenterCrop((self.h, self.w)),
                                          T.ToTensor(),
                                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])
            self.transform_mask = T.Compose([T.CenterCrop((self.h, self.w)),
                                             T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask, name = self.x[idx], self.y[idx], self.mask[idx], self.name[idx]
        # name = os.path.basename(x)
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)


        if y == 0:
            mask = torch.zeros([1, self.h, self.w])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask, name

    def get_size(self):
        return self.h, self.w

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, name = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            name_list = os.listdir(img_type_dir)
            name.extend(name_list)

            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png') or f.endswith('.bmp')])

            x.extend(img_fpath_list)


            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask), list(name)



