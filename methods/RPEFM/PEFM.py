from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import math
from methods.model import CoverModel
import numpy as np
from scipy.ndimage import gaussian_filter
from methods.de_resnet import *
from methods.resnet import *
import torch
from torch.nn import functional as F


# =============================================================
def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


# =============================================================
class Conv_BN_Relu(nn.Module):
    def __init__(self, in_dim, out_dim, k=1, s=1, p=0, bn=True, relu=True):
        super(Conv_BN_Relu, self).__init__()
        self.conv = [
            nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
        ]
        if bn:
            self.conv.append(nn.BatchNorm2d(out_dim))
        if relu:
            self.conv.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


# =============================================================
class Conv_BN_PRelu(nn.Module):
    def __init__(self, in_dim, out_dim, k=1, s=1, p=0, bn=True, prelu=True):
        super(Conv_BN_PRelu, self).__init__()
        self.conv = [
            nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
        ]
        if bn:
            self.conv.append(nn.BatchNorm2d(out_dim))
        if prelu:
            self.conv.append(nn.PReLU())

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


# =============================================================
class NonLocalAttention(nn.Module):
    def __init__(self, channel=256, reduction=2, rescale=1.0):
        super(NonLocalAttention, self).__init__()
        # self.conv_match1 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        # self.conv_match2 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        # self.conv_assembly = common.BasicBlock(conv, channel, channel, 1,bn=False, act=nn.PReLU())

        self.conv_match1 = Conv_BN_PRelu(channel, channel // reduction, 1, bn=False, prelu=True)
        self.conv_match2 = Conv_BN_PRelu(channel, channel // reduction, 1, bn=False, prelu=True)
        self.conv_assembly = Conv_BN_PRelu(channel, channel, 1, bn=False, prelu=True)
        self.rescale = rescale

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
        x_embed_2 = x_embed_2.view(N, C, H * W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N, -1, H * W).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly)
        x_final = x_final.permute(0, 2, 1).view(N, -1, H, W)
        return x_final + input * self.rescale


# =============================================================
class DualProjectionWithPENet(nn.Module):
    def __init__(self, H, W, in_dim=512, out_dim=512, latent_dim=256, dual_type="middle", pe_required=True):
        super(DualProjectionWithPENet, self).__init__()

        self.H = H
        self.W = W
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.pe_required = pe_required
        if self.pe_required:
            logger.info(f"Position Encoding is Used!")
            self.pe1 = positionalencoding2d(self.in_dim, H, W)
            self.pe1 = self.pe1.cuda()
            self.pe2 = positionalencoding2d(self.out_dim, H, W)
            self.pe2 = self.pe2.cuda()
        else:
            self.pe1 = torch.zeros([1]).cuda()
            self.pe2 = torch.zeros([1]).cuda()

        self.dual_type = dual_type
        # assert self.dual_type in ["less", "small", "middle", "large"]

        if self.dual_type == "small":
            logger.info(f"small model is Used!")
            self.encoder1 = nn.Sequential(*[
                Conv_BN_Relu(in_dim, in_dim // 2 + latent_dim),
                Conv_BN_Relu(in_dim // 2 + latent_dim, 2 * latent_dim),
                # Conv_BN_Relu(2*latent_dim, latent_dim),
            ])

            self.shared_coder = Conv_BN_Relu(2 * latent_dim, latent_dim, bn=False, relu=False)

            self.decoder1 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, 2 * latent_dim),
                Conv_BN_Relu(2 * latent_dim, out_dim // 2 + latent_dim),
                Conv_BN_Relu(out_dim // 2 + latent_dim, out_dim, bn=False, relu=False),
            ])

            self.encoder2 = nn.Sequential(*[
                Conv_BN_Relu(out_dim, out_dim // 2 + latent_dim),
                Conv_BN_Relu(out_dim // 2 + latent_dim, 2 * latent_dim),
                # Conv_BN_Relu(2 * latent_dim, latent_dim),
            ])

            self.decoder2 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, 2 * latent_dim),
                Conv_BN_Relu(2 * latent_dim, in_dim // 2 + latent_dim),
                Conv_BN_Relu(in_dim // 2 + latent_dim, in_dim, bn=False, relu=False),
            ])

        elif self.dual_type == "small_nonlocal":
            logger.info(f"small_nonlocal model is Used!")
            self.encoder1 = nn.Sequential(*[
                Conv_BN_Relu(in_dim, in_dim // 2 + latent_dim),
                Conv_BN_Relu(in_dim // 2 + latent_dim, 2 * latent_dim),
                # Conv_BN_Relu(2*latent_dim, latent_dim),
                NonLocalAttention(channel=2 * latent_dim)
            ])

            self.shared_coder = Conv_BN_Relu(2 * latent_dim, latent_dim, bn=False, relu=False)

            self.decoder1 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, 2 * latent_dim),
                NonLocalAttention(channel=2 * latent_dim),
                Conv_BN_Relu(2 * latent_dim, out_dim // 2 + latent_dim),
                Conv_BN_Relu(out_dim // 2 + latent_dim, out_dim, bn=False, relu=False),
            ])

            self.encoder2 = nn.Sequential(*[
                Conv_BN_Relu(out_dim, out_dim // 2 + latent_dim),
                Conv_BN_Relu(out_dim // 2 + latent_dim, 2 * latent_dim),
                NonLocalAttention(channel=2 * latent_dim)
            ])

            self.decoder2 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, 2 * latent_dim),
                NonLocalAttention(channel=2 * latent_dim),
                Conv_BN_Relu(2 * latent_dim, in_dim // 2 + latent_dim),
                Conv_BN_Relu(in_dim // 2 + latent_dim, in_dim, bn=False, relu=False),
            ])

        elif self.dual_type == "less":
            logger.info(f"less model is Used!")
            self.encoder1 = nn.Sequential(*[
                Conv_BN_Relu(in_dim, latent_dim)
            ])
            self.shared_coder = Conv_BN_Relu(latent_dim, latent_dim, bn=True, relu=True)
            self.decoder1 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, out_dim, bn=False, relu=False),
            ])

            self.encoder2 = nn.Sequential(*[
                Conv_BN_Relu(out_dim, latent_dim),
            ])

            self.decoder2 = nn.Sequential(*[
                Conv_BN_Relu(latent_dim, in_dim, bn=False, relu=False),
            ])

    def forward(self, xs, xt):
        xt_hat = self.encoder1(xs + self.pe1.unsqueeze(0))
        xt_hat = self.shared_coder(xt_hat)
        xt_hat = self.decoder1(xt_hat)

        xs_hat = self.encoder2(xt + self.pe2.unsqueeze(0))
        xs_hat = self.shared_coder(xs_hat)
        xs_hat = self.decoder2(xs_hat)

        return xs_hat, xt_hat


class PEFM(nn.Module):
    def __init__(self, **kwargs):
        super(PEFM, self).__init__()

        dual_type = kwargs['dual_type']

        assert dual_type in ["less", "small", "small_nonlocal", "middle", "large"]

        self.out_size_h = kwargs['out_size_h']
        self.out_size_w = kwargs['out_size_w']

        self.H2, self.W2 = self.out_size_h // 4, self.out_size_w // 4
        self.H3, self.W3 = self.out_size_h // 8, self.out_size_w // 8
        self.H4, self.W4 = self.out_size_h // 16, self.out_size_w // 16

        self.device = kwargs['device']

        self.model = self.get_model(**kwargs)

    def get_model(self, **kwargs):
        self.backbone = kwargs['backbone']

        self.dual_type = kwargs['dual_type']
        self.pe_required = kwargs['pe_required']

        AgentT, bn = eval(f'{self.backbone}(pretrained=True)')
        # However, we do not have the pre-trained parameters for de-resnet...
        AgentS = eval(f'de_{self.backbone}(pretrained=True)')

        if self.backbone in ["resnet18", "resnet34"]:
            self.dims = [64, 128, 256]
        elif self.backbone in ["resnet50", "resnet101"]:
            self.dims = [256, 512, 1024]

        self.latent_dim = [200, 400, 800]

        logger.info(f"{self.backbone}-{self.dims}-{self.latent_dim}")

        projector2 = DualProjectionWithPENet(self.H2, self.W2, in_dim=self.dims[0], out_dim=self.dims[0],
                                                  latent_dim=self.latent_dim[0], dual_type=self.dual_type,
                                                  pe_required=self.pe_required)
        logger.info(
            f"Projector2: {self.H3}, {self.W3}, in_dim={self.dims[1]}, out_dim={self.dims[1]}, "
            f"latent_dim={self.latent_dim[1]}, dual_type={self.dual_type}, pe_required={self.pe_required}")
        projector3 = DualProjectionWithPENet(self.H3, self.W3, in_dim=self.dims[1], out_dim=self.dims[1],
                                                  latent_dim=self.latent_dim[1], dual_type=self.dual_type,
                                                  pe_required=self.pe_required)
        logger.info(
            f"Projector2: {self.H4}, {self.W4}, in_dim={self.dims[2]}, out_dim={self.dims[2]}, "
            f"latent_dim={self.latent_dim[2]}, dual_type={self.dual_type}, pe_required={self.pe_required}")
        projector4 = DualProjectionWithPENet(self.H4, self.W4, in_dim=self.dims[2], out_dim=self.dims[2],
                                                  latent_dim=self.latent_dim[2], dual_type=self.dual_type,
                                                  pe_required=self.pe_required)

        for param in AgentS.parameters():
            param.requires_grad = False
        for param in AgentT.parameters():
            param.requires_grad = False

        AgentS.eval()
        AgentT.eval()

        Projector = torch.nn.ModuleList([
            projector2,
            projector3,
            projector4
        ])

        model = torch.nn.ModuleDict({'AgentS': AgentS, 'AgentT': AgentT, 'bn':bn, 'Projector': Projector})

        return model

    def forward(self, x, **kwargs)->dict:

        with torch.no_grad():
            out_a1 = self.model['AgentT'](x)

        # with grad
        embedding = self.model['bn'](out_a1)

        with torch.no_grad():
            out_a2 = self.model['AgentS'](embedding)

        project_out1_list = []
        project_out2_list = []

        for f1, f2, projector in zip(out_a1, out_a2, self.model['Projector']):
            project_out1, project_out2 = projector(f1.detach(), f2.detach())

            project_out1_list.append(project_out1)
            project_out2_list.append(project_out2)

        return {'o1':out_a1, 'o2':out_a2, 'p1':project_out1_list, 'p2':project_out2_list}

    def cal_loss(self, **kwargs) -> torch.Tensor:
        out_a1 = kwargs['o1']
        out_a2 = kwargs['o2']
        project_out1 = kwargs['p1']
        project_out2 = kwargs['p2']

        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for a1, a2, pa1, pa2 in zip(out_a1, out_a2, project_out1, project_out2):
            loss += torch.mean(1 - cos_loss(a1.view(a1.shape[0], -1),
                                            pa1.view(pa1.shape[0], -1)))
            loss += torch.mean(1 - cos_loss(a2.view(a2.shape[0], -1),
                                            pa2.view(pa2.shape[0], -1)))

        return loss

    @torch.no_grad()
    def cal_am(self, **kwargs) -> np.ndarray:
        out_a1 = kwargs['o1']
        out_a2 = kwargs['o2']
        project_out1 = kwargs['p1']
        project_out2 = kwargs['p2']

        anomaly_map = 0

        for a1, a2, pa1, pa2 in zip(out_a1, out_a2, project_out1, project_out2):
            a_map_1 = 1 - F.cosine_similarity(a1, pa1)
            a_map_1 = torch.unsqueeze(a_map_1, dim=1)
            a_map_1 = F.interpolate(a_map_1, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)

            a_map_2 = 1 - F.cosine_similarity(a2, pa2)
            a_map_2 = torch.unsqueeze(a_map_2, dim=1)
            a_map_2 = F.interpolate(a_map_2, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                    align_corners=False)

            anomaly_map += (a_map_1 + a_map_2)

        anomaly_map = anomaly_map.squeeze(1).cpu().numpy()
        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

        return anomaly_map

    def save(self, path):
        torch.save(self.model['Projector'].state_dict(), path)

    def load(self, path):
        self.model['Projector'].load_state_dict(torch.load(path, map_location=self.device))

    def train_mode(self):
        self.model['Projector'].train()

    def eval_mode(self):
        self.model['Projector'].eval()

if __name__ == '__main__':
    model = PEFM(backbone='resnet50', dual_type='small',out_size_h=256,out_size_w=256,device='cuda:0',pe_required=True)
    model.cuda()
    x = torch.rand((1,3,256,256)).to('cuda:0')

    a = model(x)

    print(a)