import torch.optim as optim
from mmdet.auto_encoder import MultiAutoEncoder
from mmdet.datasets import CoderDataset
from mmdet.datasets import build_dataloader
from mmcv.runner import save_checkpoint
import getpass


def adjust_learning_rate(optimizer,base_lr, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = base_lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
    	param_group['lr'] = lr

def main():
    # base configs
    data_root = '/media/' + getpass.getuser() + '/Data/DoubleCircle/datasets/kaist-rgb-t-eccoder'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    img_norm_cfg_t = dict(
        mean=[123.675, 123.675, 123.675], std=[58.395, 58.395, 58.395], to_rgb=False)
    imgs_per_gpu = 4
    workers_per_gpu = 2
    max_epoch = 100
    base_lr = 0.001


    # train and test dataset
    train = dict(
        ann_file=data_root + 'annotations-pkl/train-all.pkl',
        img_prefix=data_root + 'images/',
        img_scale=1.5,
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True)
    test = dict(
        ann_file=data_root + 'annotations-pkl/test-all.pkl',
        img_prefix=data_root + 'images/',
        img_scale=1.5,
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True)
    dataset_train = CoderDataset(**train)
    dataset_test = CoderDataset(**test)

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

    # train
    net = MultiAutoEncoder()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    for e in range(max_epoch):
        # training phase
        net.train()
        for i,data_batch in enumerate(data_loaders_train):
            code,decode_rgb,decode_thermal = net(**data_batch)
            loss = net.loss(img_rgb=data_batch['img_rgb'],img_thermal=data_batch['img_thermal'],
                            decode_rgb = decode_rgb, decode_thermal = decode_thermal)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {},Training Loss{}'.format(e,loss))
        # update learn rate
        adjust_learning_rate(optimizer,base_lr,e)
        # save checkpoint
        filename = '../../work_dirs/autoencoder/epoch_{}.pth'.format(e)
        save_checkpoint(net,filename=filename)
        # evaluation phase
        if (e+1) % 20==0:
            net.eval()
            for i, data_batch in enumerate(data_loaders_test):
                code, decode_rgb, decode_thermal = net(**data_batch)
                loss = net.loss(img_rgb=data_batch['img_rgb'], img_thermal=data_batch['img_thermal'],
                                decode_rgb=decode_rgb, decode_thermal=decode_thermal)
                print('Epoch {},Evaluation Loss{}\n'.format(e,loss))