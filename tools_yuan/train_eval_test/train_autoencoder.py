import torch.optim as optim
import torch
from mmdet.auto_encoder import MultiAutoEncoder
from mmdet.auto_encoder import AutoEncoder
from mmdet.auto_encoder import SingleAutoEncoder
from mmdet.auto_encoder import SingleKaistAutoEncoder
from mmdet.datasets import CoderSingleDataset
from mmdet.datasets import CoderKaistDataset
from mmdet.datasets import build_dataloader
from mmcv.runner import save_checkpoint
import getpass
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torch.autograd import Variable
import os
import mmcv
from torchvision.utils import make_grid

def save_images(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def adjust_learning_rate(optimizer, base_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 40 epochs"""
    if epoch > 120:
        return
    lr = base_lr * (0.1 ** (epoch // 40))
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
    img = (img * 128) + 128  # unnormalize
    img.clamp(0, 255)
    img = torch.from_numpy(np.uint8(img.numpy()))
    return img


def to_img_minst(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x


def main():
    # base configs
    data_root = '/media/' + getpass.getuser() + '/Data/DoubleCircle/datasets/kaist-rgbt-encoder/'
    # img_norm_cfg = dict(
    #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # img_norm_cfg_t = dict(
    #     mean=[123.675, 123.675, 123.675], std=[58.395, 58.395, 58.395], to_rgb=False)
    img_norm_cfg = dict(
                        mean=[128, 128, 128], std=[128, 128, 128], to_rgb=True)
    img_norm_cfg_t = dict(
                        mean=[70, 70, 70], std=[70, 70, 70], to_rgb=False)
    imgs_per_gpu = 256
    workers_per_gpu = 2
    max_epoch = 300
    base_lr = 1e-3

    # train and test dataset
    train = dict(
        ann_file=data_root + 'annotations-pkl/train-all.pkl',
        img_prefix=data_root + 'images/',
        img_scale=0.25,
        img_norm_cfg=img_norm_cfg,
        img_norm_cfg_t=img_norm_cfg_t,
        size_divisor=None,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True)
    test = dict(
        ann_file=data_root + 'annotations-pkl/test-all-rgb.pkl',
        img_prefix=data_root + 'images/',
        img_scale=1.0,
        img_norm_cfg=img_norm_cfg,
        img_norm_cfg_t=img_norm_cfg_t,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True)
    dataset_train = CoderSingleDataset(**train)
    dataset_test = CoderKaistDataset(**test)

    # train and test data loader
    data_loaders_train = build_dataloader(
        dataset_train,
        imgs_per_gpu,
        workers_per_gpu,
        num_gpus=1,
        dist=False)
    data_loaders_test = build_dataloader(
        dataset_test,
        imgs_per_gpu,
        workers_per_gpu,
        num_gpus=1,
        dist=False)

    # MINST dataset
    # im_tfs = tfs.Compose([
    #     tfs.ToTensor(),
    #     tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化
    # ])
    #
    # train_set = MNIST('./mnist', transform=im_tfs, download=True)
    # train_data = DataLoader(train_set, batch_size=128, shuffle=True)

    # train
    net = SingleKaistAutoEncoder()
    net.init_weights()
    net.cuda()
    # loss_fn = torch.nn.MSELoss(size_average=False)
    loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    print('Start training...\n')
    for e in range(max_epoch):
        for im in data_loaders_train:
            if torch.cuda.is_available():
                input = im['img_thermal_in'].cuda()
            input = Variable(input)
            # 前向传播
            code, output = net(input)
            loss = loss_fn(output, input)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (e + 1) % 1 == 0:  # 每 1 次，将生成的图片保存一下
            print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data))
            output = output.cpu().data
            target = im['img_thermal_in']
            pic = np.zeros((output.shape[0], output.shape[2], output.shape[3], output.shape[1]), dtype=np.uint8)
            target_pic = np.zeros((output.shape[0], output.shape[2], output.shape[3], output.shape[1]),
                                  dtype=np.uint8)
            mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
            std = np.array(img_norm_cfg['std'], dtype=np.float32)
            for idx in range(imgs_per_gpu):
                img = output[idx, ...].numpy().transpose(1, 2, 0).astype(np.float32)
                pic[idx, :, :, :] = mmcv.imdenormalize(
                    img, mean=mean, std=std, to_bgr=False).astype(np.uint8)
                target_img = target[idx, ...].numpy().transpose(1, 2, 0).astype(np.float32)
                target_pic[idx, :, :, :] = mmcv.imdenormalize(
                    target_img, mean=mean, std=std, to_bgr=False).astype(np.uint8)
            if not os.path.exists('../../work_dirs/autoencoder'):
                os.mkdir('../../work_dirs/autoencoder')
            save_images(torch.from_numpy(pic.transpose((0, 3, 1, 2))),
                       '../../work_dirs/autoencoder/image_{}.png'.format(e + 1))
            save_images(torch.from_numpy(target_pic.transpose(0, 3, 1, 2)),
                       '../../work_dirs/autoencoder/target_image_{}.png'.format(e + 1))
        # update learn rate
        adjust_learning_rate(optimizer, base_lr, e)
        # save checkpoint
        filename = '../../work_dirs/autoencoder/epoch_{}.pth'.format(e + 1)
        save_checkpoint(net, filename=filename)
    # iter_epoch = len(data_loaders_train)
    # for e in range(max_epoch):
    #     # training phase
    #     net.train()
    #     loss_iter = 0.0
    #     for i, data_batch in enumerate(data_loaders_train):
    #         code, decode_rgb, decode_thermal = net(data_batch['img_rgb_in'].cuda(),
    #                                                data_batch['img_thermal_in'].cuda())
    #         loss_rgb = loss_fn(decode_rgb.cpu(), data_batch['img_rgb_out'])
    #         loss_thermal = loss_fn(decode_thermal.cpu(), data_batch['img_thermal_out'])
    #         loss = loss_rgb + loss_thermal
    #         loss_iter += loss
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if (i + 1) % 10 == 0:
    #             print('Epoch:{},[{}|{}],Learning Rate:{},Loss:{:.4f}'.format(e + 1, i + 1, len(data_loaders_train),
    #                                                                          optimizer.param_groups[0]['lr'],
    #                                                                          loss_iter))
    #             loss_iter = 0.0
    #
    #     # print('Epoch {},Training Loss:{}\n'.format(e, loss_epoch))
    #     # update learn rate
    #     adjust_learning_rate(optimizer, base_lr, e)
    #     # save checkpoint
    #     filename = '../../work_dirs/autoencoder/epoch_{}.pth'.format(e + 1)
    #     save_checkpoint(net, filename=filename)
    #     # evaluation phase
    #     if (e + 1) % 1 == 0:
    #         net.eval()
    #         loss_total = 0.0
    #         print('Start Evaluation...\n')
    #         with torch.no_grad():
    #             for i, data_batch in enumerate(data_loaders_train):
    #                 if i < 1:
    #                     code, decode_rgb, decode_thermal = net(data_batch['img_rgb_in'].cuda(),
    #                                                            data_batch['img_thermal_in'].cuda())
    #                     loss_rgb = loss_fn(data_batch['img_rgb_out'], decode_rgb.cpu())
    #                     loss_thermal = loss_fn(data_batch['img_thermal_out'], decode_thermal.cpu())
    #                     loss_total += (loss_rgb + loss_thermal)
    #                     pic_rgb = to_img(decode_rgb.cpu())
    #                     pic_thermal = to_img(decode_thermal.cpu())
    #                     save_image(pic_rgb,
    #                                '../../work_dirs/autoencoder/reconstructed/rgb_{}_{}.png'.format(e + 1, i + 1))
    #                     save_image(pic_thermal,
    #                                '../../work_dirs/autoencoder/reconstructed/thermal_{}_{}.png'.format(e + 1, i + 1))
    #                     pic_target_rgb = to_img(data_batch['img_rgb_out'])
    #                     pic_target_thermal = to_img(data_batch['img_thermal_out'])
    #                     save_image(pic_target_rgb,
    #                                '../../work_dirs/autoencoder/reconstructed/target_rgb_{}.png'.format(i + 1))
    #                     save_image(pic_target_thermal,
    #                                '../../work_dirs/autoencoder/reconstructed/target_thermal_{}.png'.format(i + 1))
    #             print('Epoch {},Evaluation Loss:{:.4f}\n'.format(e + 1, loss_total))


if __name__ == '__main__':
    main()
