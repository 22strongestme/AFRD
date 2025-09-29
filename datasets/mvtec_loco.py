import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T





CLASS_NAMES = ['juice_bottle','splicing_connectors','pushpins','breakfast_box','screw_bag']

DATASET_PATH = '../datasets/mvtec_anomaly_detection'
class MVTecDataset(Dataset):
    def __init__(self, dataset_path=None, class_name='bottle', is_train=True,
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
        self.mvtec_folder_path = '/home/tesla/mvtec_loco_anomaly_detection' #os.path.join(root_path, 'mvtec_anomaly_detection')

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

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
            self.h, self.w = x.size

            # should be times of 32
            self.h = self.h // 32 * 32
            self.w = self.w // 32 * 32

            self.transform_x = T.Compose([T.CenterCrop((self.h, self.w)),
                                          T.ToTensor(),
                                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])
            self.transform_mask = T.Compose([T.CenterCrop((self.h, self.w)),
                                             T.ToTensor()])
    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        name = os.path.basename(x)
        # name = f'{type}-{name[:-4]}'
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
        x, y, mask = [], [], []

        img_dir = os.path.join(self.mvtec_folder_path, self.class_name, phase)
        gt_dir = os.path.join(self.mvtec_folder_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = []
                for img_fname in img_fname_list:
                    # Assuming the first part of the file name indicates the subdirectory for the mask
                    subdir = img_fname.split('_')[0]
                    mask_file_path = os.path.join(gt_type_dir, subdir, '000.png')
                    gt_fpath_list.append(mask_file_path)
                mask.extend(gt_fpath_list)

        assert len(x) == len(y) and len(y) == len(mask), 'number of x, y, and mask should be same'

        return list(x), list(y), list(mask)

    # def load_dataset_folder(self):
    #     phase = 'train' if self.is_train else 'test'
    #     x, y, mask = [], [], []
    #
    #     img_dir = os.path.join(self.dataset_path, self.class_name, phase)
    #     gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
    #
    #     img_types = sorted(os.listdir(img_dir))
    #     tot_img_types = []
    #     for img_type in img_types:
    #
    #         # load images
    #         img_type_dir = os.path.join(img_dir, img_type)
    #         if not os.path.isdir(img_type_dir):
    #             continue
    #         img_fpath_list = sorted([os.path.join(img_type_dir, f)
    #                                  for f in os.listdir(img_type_dir)
    #                                  if f.endswith('.png')])
    #         x.extend(img_fpath_list)
    #         tot_img_types.extend([img_type] * len(img_fpath_list))
    #         # load gt labels
    #         if img_type == 'good':
    #             y.extend([0] * len(img_fpath_list))
    #             mask.extend([None] * len(img_fpath_list))
    #
    #         else:
    #             y.extend([1] * len(img_fpath_list))
    #             gt_type_dir = os.path.join(gt_dir, img_type)
    #             img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
    #             gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
    #                              for img_fname in img_fname_list]
    #             mask.extend(gt_fpath_list)
    #
    #     assert len(x) == len(y), 'number of x and y should be same'
    #
    #     return list(x), list(y), list(mask), tot_img_types



