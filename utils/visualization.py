import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

##
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

##
from mpl_toolkits.mplot3d import Axes3D

def plot_sample(names, imgs, scores_:dict, gts, save_folder=None):
    # get subplot number
    subplot_number = len(scores_) + 2
    total_number = len(imgs)

    nrows = 4
    ncols = subplot_number // nrows + 1

    scores = scores_.copy()
    # normarlisze anomalies
    for k,v in scores.items():
        max_value = np.max(v)
        min_value = np.min(v)

        scores[k] = (scores[k] - min_value) / max_value

    # draw gts
    mask_imgs = []
    for idx in range(total_number):
        gts_ = gts[idx]
        mask_imgs_ = imgs[idx].copy()
        mask_imgs_[gts_ > 0.5] = (255, 0, 0)
        mask_imgs.append(mask_imgs_)

    # save imgs
    for idx in range(total_number):
        plt.figure()
        plt.clf()

        plt.subplot(ncols, nrows, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title('ORI')
        plt.imshow(imgs[idx])

        plt.subplot(ncols, nrows, 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('GT')
        plt.imshow(mask_imgs[idx])

        n = 3
        for key in scores:
            plt.subplot(ncols, nrows, n)
            plt.xticks([])
            plt.yticks([])
            plt.title(key)

            # display max
            plt.imshow(scores[key][idx], cmap="jet", vmax=1,vmin=0)
            # plt.imshow(scores[key][idx], cmap="jet")
            n = n + 1

        plt.savefig(os.path.join(save_folder,f'{names[idx]}.jpg'), bbox_inches='tight')
        plt.close()
#single-plot
# def plot_sample_cv2(names, imgs, scores_:dict, gts, save_folder=None):
#     # get subplot number
#     subplot_number = len(scores_) + 2
#     total_number = len(imgs)
#
#     nrows = 4
#     ncols = subplot_number // nrows + 1
#
#     scores = scores_.copy()
#     # normarlisze anomalies
#     for k,v in scores.items():
#         max_value = np.max(v)
#         min_value = np.min(v)
#
#         scores[k] = (scores[k] - min_value) / max_value * 255
#         scores[k] = scores[k].astype(np.uint8)
#     # draw gts
#     mask_imgs = []
#     for idx in range(total_number):
#         gts_ = gts[idx]
#         mask_imgs_ = imgs[idx].copy()
#         mask_imgs_[gts_ > 0.5] = (255, 0, 0)
#         mask_imgs.append(mask_imgs_)
#
#     # save imgs
#     for idx in range(total_number):
#         cv2.imwrite(os.path.join(save_folder,f'{names[idx]}_ori.jpg'),cv2.cvtColor(imgs[idx], cv2.COLOR_RGB2BGR))
#         cv2.imwrite(os.path.join(save_folder,f'{names[idx]}_gt.jpg'),cv2.cvtColor(mask_imgs[idx], cv2.COLOR_RGB2BGR))
#
#         for key in scores:
#             heat_map = cv2.applyColorMap(scores[key][idx],cv2.COLORMAP_JET)
#             visz_map = cv2.addWeighted(heat_map,0.5, imgs[idx], 0.5, 0)
#             cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_{key}.jpg'),
#                         visz_map)

# multi-plot
def plot_sample_cv2(imgs, scores_:dict, gts, save_folder=None):
    assert len(imgs) == 6 * len(gts), "图像数量应该是分数和掩码数量的六倍"
    # get subplot number
    # subplot_number = len(scores_) + 2
    total_number = len(imgs)



    scores = scores_.copy()
    # normarlisze anomalies
    for k,v in scores.items():
        max_value = np.max(v)
        min_value = np.min(v)

        scores[k] = (scores[k] - min_value) / max_value * 255
        scores[k] = scores[k].astype(np.uint8)

        for idx in range(0, total_number, 6):
            for i in range(6):
                # 当前图像索引
                img_index = idx + i
                # 计算批次中对应的gts和scores索引
                batch_index = idx // 6

                # 保存原始图像
                cv2.imwrite(os.path.join(save_folder, f'img_{img_index}_ori.jpg'),
                            cv2.cvtColor(imgs[img_index], cv2.COLOR_RGB2BGR))

                # 保存带有掩码的图像
                mask_img = imgs[img_index].copy()
                if np.any(gts[batch_index] > 0.5):  # 假设gt为二进制掩码
                    mask_img[gts[batch_index] > 0.5] = (255, 0, 0)
                cv2.imwrite(os.path.join(save_folder, f'img_{img_index}_gt.jpg'),
                            cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

                # 对于每个分数类型，保存带有热图的图像
                for key, score_vals in scores.items():
                    heat_map = cv2.applyColorMap(score_vals[batch_index], cv2.COLORMAP_JET)
                    visz_map = cv2.addWeighted(heat_map, 0.5, imgs[img_index], 0.5, 0)
                    cv2.imwrite(os.path.join(save_folder, f'img_{img_index}_{key}.jpg'), visz_map)
    # # draw gts
    # mask_imgs = []
    # for idx in range(total_number):
    #     gts_ = gts[idx]
    #     mask_imgs_ = imgs[idx].copy()
    #     mask_imgs_[gts_ > 0.5] = (255, 0, 0)
    #     mask_imgs.append(mask_imgs_)
    #
    # # save imgs
    # for idx in range(total_number):
    #     cv2.imwrite(os.path.join(save_folder,f'{idx}_ori.jpg'),cv2.cvtColor(imgs[idx], cv2.COLOR_RGB2BGR))
    #     cv2.imwrite(os.path.join(save_folder,f'{idx}_gt.jpg'),cv2.cvtColor(mask_imgs[idx], cv2.COLOR_RGB2BGR))
    #
    #     for key in scores:
    #         heat_map = cv2.applyColorMap(scores[key][idx],cv2.COLORMAP_JET)
    #         visz_map = cv2.addWeighted(heat_map,0.5, imgs[idx], 0.5, 0)
    #         cv2.imwrite(os.path.join(save_folder, f'{idx}_{key}.jpg'),
    #                     visz_map)

def plot_anomaly_score_distributions(scores:dict, ground_truths_list, save_folder, class_name):

    ground_truths = np.stack(ground_truths_list, axis=0)

    N_COUNT = 100000

    for k, v in scores.items():

        layer_score = np.stack(v, axis=0)
        normal_score = layer_score[ground_truths == 0]
        abnormal_score = layer_score[ground_truths != 0]

        plt.clf()

        sns.kdeplot(np.random.choice(abnormal_score, N_COUNT), shade='fill', label='${d(p_a)}$', color='red')
        sns.kdeplot(np.random.choice(normal_score, N_COUNT), shade='fill', label='${d(p_n)}$', color='green')

        # sns.kdeplot(abnormal_score, shade='fill', label='${d(p_a)}$')
        # sns.kdeplot(normal_score, shade='fill', label='${d(p_n)}$')

        save_path = os.path.join(save_folder, f'0_distributions_{class_name}_{k}.jpg')

        plt.savefig(save_path, bbox_inches='tight', dpi=300)


    # for idx, (nd, sad, ad, loss) in enumerate(zip(normal_distance, s_abnormal_distance, abnormal_distance, loss_type)):
    #     plt.subplot(2, 2, idx + 1)
    #
    #     sns.kdeplot(ad, shade='fill', label='${d(p_a)}$')
    #     sns.kdeplot(nd, shade='fill', label='${d(p_n)}$')
    #
    #     if loss not in ['FocalL2', 'L2']:
    #         sns.kdeplot(sad, shade='fill', label='${d(p_s)}$')


    pass

valid_feature_visualization_methods = ['TSNE', 'PCA']

def visualize_feature(features, labels, legends, n_components=3, method='TSNE'):

    assert method in valid_feature_visualization_methods
    assert n_components in [2, 3]

    if method == 'TSNE':
        model = TSNE(n_components=n_components)
    elif method == 'PCA':
        model = PCA(n_components=n_components)

    else:
        raise NotImplementedError

    feat_proj = model.fit_transform(features)

    if n_components == 2:
        ax = scatter_2d(feat_proj, labels)
    elif n_components == 3:
        ax = scatter_3d(feat_proj, labels)
    else:
        raise NotImplementedError

    plt.legend(legends)
    plt.axis('off')



def scatter_3d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes(projection='3d')

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter3D(feat_proj[label==l, 0],
                      feat_proj[label==l, 1],
                      feat_proj[label==l, 2],s=5)

    return ax1

def scatter_2d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes()

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter(feat_proj[label==l, 0],
                      feat_proj[label==l, 1],s=5)

    return ax1