import os.path as osp
import mmcv
import numpy as np
from tools_yuan.convert_data.utils import parse_xml_cross
from tools_yuan.convert_data.utils import track_progress_yuan
import getpass

"""
Author: Yuan Yuan
Date:2019/03/08
Location:SCU
"""


def main():
    xml_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt/annotations-xml/')
    pkl_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt-cross/annotations-pkl/')
    txt_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt/imageSets/')
    img_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt/images/')
    mmcv.mkdir_or_exist(pkl_dir)

    # all images
    train_filelist = osp.join(txt_dir, 'train-all-02.txt')
    img_all_names = mmcv.list_from_file(train_filelist)
    xml_all_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_all_names]
    img_all_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                     img_all_names]
    xml_all_paths = np.array(xml_all_paths)
    img_all_paths = np.array(img_all_paths)
    flags = ['train' for _ in img_all_paths]

    total_imgs = len(img_all_names)
    permutation = np.random.permutation(total_imgs)
    base_num = int(1.0 / 3 * total_imgs)
    idx_mul = permutation[0:base_num]
    idx_rgb = permutation[base_num:base_num * 2]
    idx_thermal = permutation[base_num * 2:]

    flags_model = [0 for _ in img_all_names]
    for idx in idx_mul:
        flags_model[idx] = 0
    for idx in idx_rgb:
        flags_model[idx] = 1
    for idx in idx_thermal:
        flags_model[idx] = 2

    xml_train_paths = xml_all_paths
    img_train_paths = img_all_paths
    train_annotations = track_progress_yuan(parse_xml_cross,
                                            list(zip(xml_train_paths, img_train_paths, flags, flags_model)))
    mmcv.dump(train_annotations, osp.join(pkl_dir, 'train-all.pkl'))

    # all test images
    test_filelist = osp.join(txt_dir, 'test-all-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                      img_test_names]
    flags = ['test' for _ in img_test_paths]
    # test in RGB model
    flags_model = [1 for _ in img_test_paths]
    test_annotations = track_progress_yuan(parse_xml_cross,
                                           list(zip(xml_test_paths, img_test_paths, flags, flags_model)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-all-rgb.pkl'))

    # test in Thermal model
    flags_model = [2 for _ in img_test_paths]
    test_annotations = track_progress_yuan(parse_xml_cross,
                                           list(zip(xml_test_paths, img_test_paths, flags, flags_model)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-all-thermal.pkl'))


if __name__ == '__main__':
    main()
