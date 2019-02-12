import mmcv
import numpy as np
import os.path as osp
from tools_yuan.convert_data.voc_to_coco import CoCoData
import getpass

"""
Author: Yuan Yuan
Date:2018/12/16
Location:SCU
"""


def main():
    xml_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt/annotations-xml-en/')
    json_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt/annotations-json/')
    txt_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt/imageSets/')
    img_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt/images/')
    mmcv.mkdir_or_exist(json_dir)

    # all train images
    train_filelist = osp.join(txt_dir, 'train-all-02.txt')
    img_all_names = mmcv.list_from_file(train_filelist)
    xml_all_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_all_names]
    img_all_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                     img_all_names]
    xml_all_paths = np.array(xml_all_paths)
    img_all_paths = np.array(img_all_paths)
    # total_imgs = len(img_all_names)ddddddddd
    # permutation = np.random.permutation(total_imgs)
    # num_train = int(total_imgs * 0.9)  # ratio used to train:0.9

    # train images
    # idx_train = permutation[0:num_train+1]
    xml_train_paths = xml_all_paths
    img_train_paths = img_all_paths
    coco_train = CoCoData(xml_train_paths, img_train_paths, osp.join(json_dir, 'train-all.json'))
    coco_train.convert()

    # validation images
    # idx_test = permutation[num_train + 1:]
    # xml_val_paths = xml_all_paths[idx_test]
    # img_val_paths = img_all_paths[idx_test]
    # coco_val = CoCoData(xml_val_paths, img_val_paths, osp.join(json_dir, 'val.json'))
    # coco_val.convert()

    # test images(all)
    test_filelist = osp.join(txt_dir, 'test-all-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                      img_all_names]
    coco_test = CoCoData(xml_test_paths, img_test_paths, osp.join(json_dir, 'test-all.json'))
    coco_test.convert()

    # test images(day)
    test_filelist = osp.join(txt_dir, 'test-day-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                      img_all_names]
    coco_test = CoCoData(xml_test_paths, img_test_paths, osp.join(json_dir, 'test-day.json'))
    coco_test.convert()

    # test images(night)
    test_filelist = osp.join(txt_dir, 'test-night-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                      img_all_names]
    coco_test = CoCoData(xml_test_paths, img_test_paths, osp.join(json_dir, 'test-night.json'))
    coco_test.convert()


if __name__ == '__main__':
    main()
