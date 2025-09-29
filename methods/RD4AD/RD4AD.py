from methods.model import CoverModel
import numpy as np
from scipy.ndimage import gaussian_filter
from methods.de_resnet import *
from methods.resnet import *
import torch.nn as nn
import torch
from torch.nn import functional as F

valid_backbones = ['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2']


class AttentionModule(nn.Module):
    def __init__(self, channel, height, width):
        super(AttentionModule, self).__init__()
        self.fc = nn.Linear(channel * height * width, 1).to("cuda:0")

    def forward(self, features):
        device = features[0].device
        # features is a list of tensors with shape [batch_size, C, H, W]
        batch_size, C, H, W = features[0].shape
        num_images = len(features)

        # Flatten features for attention
        feature_stack = torch.stack(features, dim=1).to(device)
        # feature_stack_flatten = feature_stack.view(batch_size, num_images, -1)
        feature_stack_flatten = feature_stack.reshape(batch_size, num_images, -1)

        attention_weights = self.fc(feature_stack_flatten)
        attention_weights = F.softmax(attention_weights.view(batch_size, num_images), dim=1)  # [batch_size, num_images]
        attention_weights = attention_weights.view(batch_size, num_images, 1, 1, 1)

        attention_weights = attention_weights.expand(-1, -1, C, H, W)

        feature_stack_weighted = feature_stack * attention_weights
        fused_feature = feature_stack_weighted.sum(dim=1)

        return fused_feature




class RD4AD(CoverModel):
    def __init__(self, **kwargs):
        super(RD4AD, self).__init__()
        self.out_size_h = kwargs['out_size_h']
        self.out_size_w = kwargs['out_size_w']

        self.device = kwargs['device']

        self.model = self.get_model(**kwargs)
        self.attention_modules = nn.ModuleList([
            # AttentionModule(768, 64, 64),
            # AttentionModule(768, 32, 32),
            # AttentionModule(768, 16, 16)
            AttentionModule(256, 64, 64),
            AttentionModule(512, 32, 32),
            AttentionModule(1024, 16, 16)
        ])


    def get_model(self, **kwargs) ->torch.nn.Module:
        backbone = kwargs['backbone']

        assert backbone in valid_backbones, f"We only support backbones in {valid_backbones}"

        encoder, bn = eval(f'{backbone}(pretrained=True)')
        decoder = eval(f'de_{backbone}(pretrained=False)')

        model_t = torch.nn.Sequential(*[
            encoder
        ])

        model_s = torch.nn.Sequential(*[
            bn,
            decoder
        ])

        for param in model_t.parameters():
            param.requires_grad = False

        model_t.eval()

        model = torch.nn.ModuleDict({'t':model_t, 's':model_s})

        return model

    # def forward(self, x, **kwargs)->dict:
    #
    #     with torch.no_grad():
    #         feature_t = self.model['t'](x)
    #     feature_s = self.model['s'](feature_t)
    #
    #     return {'ft':feature_t, 'fs':feature_s}


    def forward(self, x_list, **kwargs) -> dict:
        x_list=x_list.to(self.device)
        # x_list is a batch of size [batch_size, 6, channels, height, width]
        batch_size, num_images, c, h, w = x_list.size()

        feature_t_list = [[] for _ in range(3)]
        for i in range(num_images):
            x = x_list[:, i, :, :, :].to(self.device)
            with torch.no_grad():
                features = self.model['t'](x)
            for j, feature in enumerate(features):
                feature_t_list[j].append(feature.to(self.device))

        fused_feature_t = []
        for i, features in enumerate(feature_t_list):
            attention_module = self.attention_modules[i]
            fused_feature = attention_module(features)
            fused_feature_t.append(fused_feature)


        feature_s = self.model['s'](fused_feature_t)

        return {'ft': fused_feature_t, 'fs': feature_s}




    def cal_loss(self, **kwargs) ->torch.Tensor:
        ft_list = kwargs['ft']
        fs_list = kwargs['fs']

        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(ft_list)):
            loss += torch.mean(1 - cos_loss(ft_list[item].view(ft_list[item].shape[0], -1),
                                            fs_list[item].view(fs_list[item].shape[0], -1)))
        return loss

    @torch.no_grad()
    def cal_am(self, **kwargs)->np.ndarray:
        ft_list = kwargs['ft']
        fs_list = kwargs['fs']

        anomaly_map = 0
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            _, _, h, w = fs.shape
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)
            anomaly_map += a_map
        anomaly_map = anomaly_map.squeeze(1).cpu().numpy()
        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

        return anomaly_map

    def save(self, path):
        torch.save(self.model['s'].state_dict(), path)

    def load(self, path):
        self.model['s'].load_state_dict(torch.load(path, map_location=self.device))

    def train_mode(self):
        self.model['t'].eval()
        self.model['s'].train()

    def eval_mode(self):
        self.model['t'].eval()
        self.model['s'].eval()

