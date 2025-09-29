import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Module
import os
import argparse
import time
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import *
from methods import *
from methods.model import CoverModel
from utils.visualization import *
from utils.metrics import *
from utils.visualization import *
from loguru import logger
from config import config

def get_optimizer_from_args(model, lr, weight_decay, **kwargs)->torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                      weight_decay=weight_decay)

def get_dir_from_args(root_dir, method, class_name, backbone, **kwargs):
    exp_name = f'{method}-{backbone}'
    model_dir = os.path.join(root_dir, exp_name, 'models')
    img_dir = os.path.join(root_dir, exp_name, 'imgs', class_name)
    tensorboard_dir = os.path.join(root_dir, exp_name, 'tensorboard', class_name)
    logger_dir = os.path.join(root_dir, exp_name, 'logger', class_name)

    log_file_name = os.path.join(logger_dir,
                                 f'log_{time.strftime("%Y-%m-%d-%H-%I-%S", time.localtime(time.time()))}.log')

    model_name = f'{class_name}'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)

    logger.start(log_file_name)

    logger.info(f"===> Root dir for this experiment: {logger_dir}")


    return model_dir, img_dir, tensorboard_dir, logger_dir, model_name

def get_tensorboard_logger_from_args(tensorboard_dir, reset_version=False):
    if reset_version:
        shutil.rmtree(os.path.join(tensorboard_dir))
    return SummaryWriter(log_dir=tensorboard_dir)

def train_epoch(model:CoverModel, dataloader:DataLoader, optimizer:torch.optim.Optimizer, device:str, **kwargs):
    model.train_mode()
    loss_sum = 0
    for (data, _, _, _) in dataloader:
        data = data.to(device)
        outputs = model(data)
        loss = model.cal_loss(**outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
    return loss_sum

def test_epoch(model:CoverModel, dataloader:DataLoader, device:str, is_vis, img_dir, class_name, cal_pro, **kwargs):
    model.eval_mode()

    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    for (data, label, mask, name) in dataloader:
        for d, n in zip(data, name):
            test_imgs.append(denormalization(d.cpu().numpy()))
            names.append(n)

        gt_list.extend(label.cpu().numpy())
        for i in range(mask.shape[0]):
            gt_mask_list.append(mask[i].squeeze().cpu().numpy())

        data = data.to(device)
        outputs = model(data)
        score = model.cal_am(**outputs)
        for i in range(score.shape[0]):
            scores.append(score[i])

    img_roc_auc, per_pixel_rocauc, pro_auc_score, threshold = \
        metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro)

    if is_vis:
        plot_sample_cv2(names, test_imgs, {'sum':scores}, gt_mask_list, save_folder=img_dir)
    #     plot_anomaly_score_distributions( {'sum':scores}, gt_mask_list, save_folder=img_dir,
    #                                      class_name=class_name)

    return {'img_auroc':img_roc_auc*100, 'pxl_auroc':per_pixel_rocauc*100, 'pxl_pro':pro_auc_score*100, 'threshold':threshold}


def main(args):
    kwargs = vars(args)

    device = f"cuda:{kwargs['gpu_id']}"
    kwargs['device'] = device
    # dataset='../datasets/multi_lighting/separate_lighting4'
    dataset = 'E:/DATASET/multi_lighting/separate_lighting4'
    # dataset='../datasets/multi_lighting/all_lighting'

    # 路径
    model_dir, img_dir, tensorboard_dir, logger_dir, model_name = get_dir_from_args(**kwargs)
    logger.info(f"===> Dataset dir for this experiment: {dataset}")

    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train',dataset_path=dataset,**kwargs)
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test',dataset_path=dataset,**kwargs)

    h, w = train_dataset_inst.get_size()
    kwargs['out_size_h'] = h
    kwargs['out_size_w'] = w

    # 获取模型
    model = get_model_from_args(**kwargs)
    model = model.to(device)

    # 获取优化器
    optimizer = get_optimizer_from_args(model=model, weight_decay=0.0001, **kwargs)

    # 获取tensorboard writter
    tensorboard_logger = get_tensorboard_logger_from_args(tensorboard_dir, True)

    epoch_bar = tqdm(range(kwargs['num_epochs']), desc=f"{kwargs['method']}:{kwargs['class_name']}")

    for epoch in epoch_bar:

        loss_sum = train_epoch(model, train_dataloader, optimizer, device)
        tensorboard_logger.add_scalar('loss', loss_sum, epoch)

        if epoch % kwargs['validation_epoch'] == 0:

            # if (kwargs['num_epochs'] - epoch) <= kwargs['validation_epoch']:
            if epoch >= kwargs['validation_epoch']:
                is_viz = kwargs['vis']
            else:
                is_viz = False

            model.save(os.path.join(model_dir, f'{model_name}_{epoch}.pt'))
            metrics = test_epoch(model, test_dataloader, device, is_viz, img_dir,
                                       class_name=kwargs['class_name'], cal_pro=False)

            logger.info(f"\n")
            for k, v in metrics.items():
                tensorboard_logger.add_scalar(f'{k}', v, epoch)
                logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

#Marshmallow  PeppermintCandy
def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    # parser.add_argument('--phase', default='test')
    parser.add_argument('--dataset', type=str, default='candies', choices=['candies', 'mvtec'])
    # parser.add_argument('--dataset_dir', type=str, default='../datasets/multi_lighting/all_lighting')
    parser.add_argument('--class_name', type=str, default='Marshmallow')
    # parser.add_argument('--class_name', type=str, default='transistor')
    parser.add_argument('--img_resize', type=int, default=256)
    parser.add_argument('--img_cropsize', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root_dir", type=str, default="./result")
    parser.add_argument("--method", type=str, default="RD4AD", choices=['RD4AD', 'ST', 'PEFM'])
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2'])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--do_resize", type=str2bool, default=True)
    parser.add_argument("--validation_epoch", type=int, default=10)

    args = parser.parse_args()

    return args

def load_config(from_dict=True):
    args = get_args()

    if from_dict:
        if config.get(args.method) is not None:
            config_dict = config[args.method]
            for k, v in config_dict.items():
                args.__dict__.update({k:v})
    return args

# NOTE: 进入tmux后再运行代码，可使代码在服务器后台运行
# NOTE：可编写sh（shell）文件，传入若干参数，方便快捷地运行代码
if __name__ == '__main__':

    args = load_config(from_dict=True)
    main(args)