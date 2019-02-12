import os.path as osp
import mmcv
import numpy as np
from tools_yuan.convert_data.utils import parse_xml
from tools_yuan.convert_data.utils import track_progress_yuan
import getpass

"""
Author: Yuan Yuan
Date:2018/12/16
Location:SCU
"""


def main():
    xml_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt/annotations-xml-en/')
    pkl_dir = osp.join('/media/', getpass.getuser(), 'Data/DoubleCircle/datasets/kaist-rgbt/annotations-pkl/')
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
    # total_imgs = len(img_all_names)
    # permutation = np.random.permutation(total_imgs)
    # num_train = int(total_imgs * 0.9)  # ratio used to train:0.9

    # train images
    # idx_train = permutation[0:num_train + 1]
    xml_train_paths = xml_all_paths
    img_train_paths = img_all_paths
    train_annotations = track_progress_yuan(parse_xml,
                                            list(zip(xml_train_paths, img_train_paths)))
    mmcv.dump(train_annotations, osp.join(pkl_dir, 'train-all.pkl'))

    # validation images
    # idx_test = permutation[num_train + 1:]
    # xml_val_paths = xml_all_paths[idx_test]
    # img_val_paths = img_all_paths[idx_test]
    # val_annotations = mmcv.track_progress(parse_xml,
    #                                       list(zip(xml_val_paths, img_val_paths)))
    # mmcv.dump(val_annotations, osp.join(pkl_dir, 'val.pkl'))

    # all test images
    test_filelist = osp.join(txt_dir, 'test-all-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                      img_all_names]
    test_annotations = track_progress_yuan(parse_xml,
                                           list(zip(xml_test_paths, img_test_paths)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-all.pkl'))

    # day test images
    test_filelist = osp.join(txt_dir, 'test-day-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                      img_all_names]
    test_annotations = track_progress_yuan(parse_xml,
                                           list(zip(xml_test_paths, img_test_paths)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-day.pkl'))

    # night test images
    test_filelist = osp.join(txt_dir, 'test-night-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                      img_all_names]
    test_annotations = track_progress_yuan(parse_xml,
                                           list(zip(xml_test_paths, img_test_paths)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-night.pkl'))


if __name__ == '__main__':
    main()
