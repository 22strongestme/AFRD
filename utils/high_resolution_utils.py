import torch
import numpy as np

def image2patch(img, patch_sz=512, overlap_sz=128)->(torch.Tensor, list, list):
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 1

    H, W = img.shape[2:]

    xs = np.arange(0, W - patch_sz, patch_sz - overlap_sz)
    ys = np.arange(0, H - patch_sz, patch_sz - overlap_sz)

    xs = np.concatenate([xs, [W - patch_sz]])
    ys = np.concatenate([ys, [H - patch_sz]])
    # xs[-1] = W - patch_sz
    # ys[-1] = H - patch_sz

    patch_list = []
    local_posi_list = []
    global_posi_list = []

    for y in ys:
        for x in xs:
            patch = img[:, :, y:y+patch_sz, x:x+patch_sz]

            local_posi = [0, 0, patch_sz, patch_sz]
            global_posi = [x, y, x + patch_sz, y + patch_sz]

            local_posi[0] = local_posi[0] + overlap_sz // 2 if global_posi[0] > 0 else local_posi[0]
            local_posi[1] = local_posi[1] + overlap_sz // 2 if global_posi[1] > 0 else local_posi[1]
            local_posi[2] = local_posi[2] - overlap_sz // 2 if global_posi[2] < W else local_posi[2]
            local_posi[3] = local_posi[3] - overlap_sz // 2 if global_posi[3] < H else local_posi[3]

            global_posi[0] = global_posi[0] + overlap_sz // 2 if global_posi[0] > 0 else global_posi[0]
            global_posi[1] = global_posi[1] + overlap_sz // 2 if global_posi[1] > 0 else global_posi[1]
            global_posi[2] = global_posi[2] - overlap_sz // 2 if global_posi[2] < W else global_posi[2]
            global_posi[3] = global_posi[3] - overlap_sz // 2 if global_posi[3] < H else global_posi[3]

            patch_list.append(patch)
            local_posi_list.append(local_posi)
            global_posi_list.append(global_posi)

    return torch.cat(patch_list, dim=0), local_posi_list, global_posi_list

def patch2image(patches, local_posi, global_posi, H, W)->torch.Tensor:

    assert isinstance(patches, torch.Tensor)

    C = patches.shape[1]

    out_img = torch.zeros((1, C, H, W)).to(patches.device)

    for patch, lo_p, glo_p in zip(patches, local_posi, global_posi):
        patch = patch.unsqueeze(0)
        out_img[:, :, glo_p[1]:glo_p[3], glo_p[0]:glo_p[2]] = patch[:, :, lo_p[1]:lo_p[3], lo_p[0]:lo_p[2]]

    return out_img
