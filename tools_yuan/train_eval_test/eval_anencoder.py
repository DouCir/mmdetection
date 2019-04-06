import torch
from mmdet.auto_encoder import AutoEncoder
from mmdet.datasets import CoderKaistDataset
from mmdet.datasets import build_dataloader
import getpass
import numpy as np
from mmcv.runner import load_checkpoint
import mmcv
import os
from torchvision.utils import make_grid


def save_images(tensor, filename, nrow=8, padding=2,
                normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def main():
    # base configs
    data_root = '/media/' + getpass.getuser() + '/Data/DoubleCircle/datasets/kaist-rgbt-encoder/'
    # img_norm_cfg = dict(
    #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # img_norm_cfg_t = dict(
    #     mean=[123.675, 123.675, 123.675], std=[58.395, 58.395, 58.395], to_rgb=False)
    img_norm_cfg = dict(
        mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
    img_norm_cfg_t = dict(
        mean=[0, 0, 0], std=[147, 147, 147], to_rgb=False)
    imgs_per_gpu = 4
    workers_per_gpu = 2

    # test dataset
    test = dict(
        ann_file=data_root + 'annotations-pkl/test-all-rgb.pkl',
        img_prefix=data_root + 'images/',
        img_scale=1.5,
        img_norm_cfg=img_norm_cfg,
        img_norm_cfg_t=img_norm_cfg_t,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        test_mode=True)
    dataset_test = CoderKaistDataset(**test)
    data_loaders_test = build_dataloader(
        dataset_test,
        imgs_per_gpu,
        workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)

    # train
    net = AutoEncoder()
    load_checkpoint(net, '../../work_dirs/autoencoder/epoch_50.pth')
    net.cuda()
    loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
    net.eval()
    with torch.no_grad():
        for i, data_batch in enumerate(data_loaders_test):
            code, decode_rgb, decode_thermal = net(data_batch['img_rgb_in'].cuda(),
                                                   data_batch['img_thermal_in'].cuda())

            loss_rgb = loss_fn(decode_rgb.cpu(), data_batch['img_rgb_out'])
            loss_thermal = loss_fn(decode_thermal.cpu(), data_batch['img_thermal_out'])
            loss_total = loss_thermal + loss_rgb
            print('Evaluation:[{}|{}],Loss:{}\n'.format(i, len(data_loaders_test), loss_total))

            output_rgb = decode_rgb.cpu().data
            target_rgb = data_batch['img_rgb_out']
            output_thermal = decode_thermal.cpu().data
            tartget_thermal = data_batch['img_thermal_out']
            pic_rgb = np.zeros((output_rgb.shape[0], output_rgb.shape[2], output_rgb.shape[3], output_rgb.shape[1]),
                               dtype=np.uint8)
            target_pic_rgb = np.zeros((output_rgb.shape[0], output_rgb.shape[2], output_rgb.shape[3],
                                       output_rgb.shape[1]), dtype=np.uint8)
            pic_thermal = np.zeros((output_rgb.shape[0], output_rgb.shape[2], output_rgb.shape[3], output_rgb.shape[1]),
                                   dtype=np.uint8)
            target_pic_thermal = np.zeros((output_rgb.shape[0], output_rgb.shape[2], output_rgb.shape[3],
                                           output_rgb.shape[1]), dtype=np.uint8)
            mean_rgb = np.array(img_norm_cfg['mean'], dtype=np.float32)
            std_rgb = np.array(img_norm_cfg['std'], dtype=np.float32)
            mean_thermal = np.array(img_norm_cfg_t['mean'], dtype=np.float32)
            std_thermal = np.array(img_norm_cfg_t['std'], dtype=np.float32)
            for idx in range(output_rgb.shape[0]):
                # for rgb
                img = output_rgb[idx, ...].numpy().transpose(1, 2, 0).astype(np.float32)
                pic_rgb[idx, :, :, :] = mmcv.imdenormalize(
                    img, mean=mean_rgb, std=std_rgb, to_bgr=False).astype(np.uint8)
                target_img = target_rgb[idx, ...].numpy().transpose(1, 2, 0).astype(np.float32)
                target_pic_rgb[idx, :, :, :] = mmcv.imdenormalize(
                    target_img, mean=mean_rgb, std=std_rgb, to_bgr=False).astype(np.uint8)
                # for thermal
                img_t = output_thermal[idx, ...].numpy().transpose(1, 2, 0).astype(np.float32)
                pic_thermal[idx, :, :, :] = mmcv.imdenormalize(
                    img_t, mean=mean_thermal, std=std_thermal, to_bgr=False).astype(np.uint8)
                target_img_t = tartget_thermal[idx, ...].numpy().transpose(1, 2, 0).astype(np.float32)
                target_pic_thermal[idx, :, :, :] = mmcv.imdenormalize(
                    target_img_t, mean=mean_thermal, std=std_thermal, to_bgr=False).astype(np.uint8)
            if not os.path.exists('../../work_dirs/autoencoder/test_rgb'):
                os.mkdir('../../work_dirs/autoencoder/test_rgb')
            save_images(torch.from_numpy(pic_rgb.transpose((0, 3, 1, 2))),
                        '../../work_dirs/autoencoder/test_rgb/image_rgb_{}.png'.format(i + 1))
            save_images(torch.from_numpy(target_pic_rgb.transpose(0, 3, 1, 2)),
                        '../../work_dirs/autoencoder/test_rgb/target_image_rgb_{}.png'.format(i + 1))
            save_images(torch.from_numpy(pic_thermal.transpose((0, 3, 1, 2))),
                        '../../work_dirs/autoencoder/test_rgb/image_thermal_{}.png'.format(i + 1))
            save_images(torch.from_numpy(target_pic_thermal.transpose(0, 3, 1, 2)),
                        '../../work_dirs/autoencoder/test_rgb/target_image_thermal_{}.png'.format(i + 1))


if __name__ == '__main__':
    main()
