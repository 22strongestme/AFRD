from .candies import CustomDataset
import torch
import numpy as np
from loguru import logger

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(dataset, class_name, img_resize, img_cropsize, batch_size, dataset_path=None,phase='train', do_resize=True, **kwargs)->torch.utils.data.DataLoader:
    ###
    if phase == 'train':
        is_train = True
        is_shuffle = True
    else:
        is_train = False
        is_shuffle = False

    if dataset == 'candies':

        dataset_inst = CustomDataset(class_name=class_name, is_train=is_train, resize=img_resize,
                                          cropsize=img_cropsize, dataset_path=dataset_path)


    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(dataset_inst, batch_size=batch_size, shuffle=is_shuffle)
    size = dataset_inst.get_size()
    debug_str = f'===> dataset: {dataset}, class name/len: {class_name}/{len(dataset_inst)}, size: {size}, batch size: {batch_size}'
    logger.info(debug_str)

    return data_loader, dataset_inst
