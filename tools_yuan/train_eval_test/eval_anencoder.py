import torch.optim as optim
import torch
from mmdet.auto_encoder import MultiAutoEncoder
from mmdet.auto_encoder import AutoEncoder,SingleKaistAutoEncoder
from mmdet.datasets import CoderKaistDataset
from mmdet.datasets import build_dataloader
from mmcv.runner import save_checkpoint
import getpass
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from mmcv.runner import load_checkpoint


def adjust_learning_rate(optimizer, base_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def imshow(img, str_title):
    img = (img * 128.0) + 128  # unnormalize
    img.clamp(0, 255)
    npimg = np.uin8(img.numpy())
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.title(str_title)


def to_img(img):
    img = (img * 128.0) + 128  # unnormalize
    img.clamp(0, 255)
    img = torch.from_numpy(np.uint8(img.numpy()))
    return img


def main():
    # base configs
    data_root = '/media/' + getpass.getuser() + '/Data/DoubleCircle/datasets/kaist-rgbt-encoder/'
    # img_norm_cfg = dict(
    #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # img_norm_cfg_t = dict(
    #     mean=[123.675, 123.675, 123.675], std=[58.395, 58.395, 58.395], to_rgb=False)
    img_norm_cfg = dict(
        mean=[128, 128, 128], std=[128, 128, 128], to_rgb=False)
    img_norm_cfg_t = dict(
        mean=[128, 128, 128], std=[128, 128, 128], to_rgb=False)
    imgs_per_gpu = 128
    workers_per_gpu = 2
    max_epoch = 100
    base_lr = 0.001

    train = dict(
        ann_file=data_root + 'annotations-pkl/train-all.pkl',
        img_prefix=data_root + 'images/',
        img_scale=0.125,
        img_norm_cfg=img_norm_cfg,
        img_norm_cfg_t=img_norm_cfg_t,
        size_divisor=None,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True)
    # test dataset
    test = dict(
        ann_file=data_root + 'annotations-pkl/test-all-rgb.pkl',
        img_prefix=data_root + 'images/',
        img_scale=0.25,
        img_norm_cfg=img_norm_cfg,
        img_norm_cfg_t=img_norm_cfg_t,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True)
    dataset_train = CoderKaistDataset(**train)
    dataset_test = CoderKaistDataset(**test)
    data_loaders_test = build_dataloader(
        dataset_test,
        imgs_per_gpu,
        workers_per_gpu,
        num_gpus=1,
        dist=False)
    data_loaders_train = build_dataloader(
        dataset_train,
        imgs_per_gpu,
        workers_per_gpu,
        num_gpus=1,
        dist=False
    )

    # train
    net = SingleKaistAutoEncoder()
    load_checkpoint(net, '../../work_dirs/autoencoder/epoch_40.pth')
    net.cuda()
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
    net.eval()
    loss_total = 0.0
    e = 9
    with torch.no_grad():
        for i, data_batch in enumerate(data_loaders_train):
            if i < 20:
                code, decode_rgb, decode_thermal = net(data_batch['img_rgb_in'].cuda(),
                                                       data_batch['img_thermal_in'].cuda())
                loss_rgb = loss_fn(data_batch['img_rgb_out'], decode_rgb.cpu())
                loss_thermal = loss_fn(data_batch['img_thermal_out'], decode_thermal.cpu())
                loss_total += (loss_rgb + loss_thermal)
                pic_rgb = to_img(decode_rgb.cpu().data)
                pic_thermal = to_img(decode_thermal.cpu().data)
                save_image(pic_rgb, '../../work_dirs/autoencoder/reconstructed/rgb_{}_{}.png'.format(e + 1, i + 1))
                save_image(pic_thermal,
                           '../../work_dirs/autoencoder/reconstructed/thermal_{}_{}.png'.format(e + 1, i + 1))

                pic_target_rgb = to_img(data_batch['img_rgb_out'])
                pic_target_thermal = to_img(data_batch['img_thermal_out'])
                save_image(pic_target_rgb, '../../work_dirs/autoencoder/reconstructed/target_rgb_{}.png'.format(i + 1))
                save_image(pic_target_thermal,
                           '../../work_dirs/autoencoder/reconstructed/target_thermal_{}.png'.format(i + 1))
        print('Epoch {},Evaluation Loss:{:.4f}\n'.format(e + 1, loss_total))


if __name__ == '__main__':
    main()
