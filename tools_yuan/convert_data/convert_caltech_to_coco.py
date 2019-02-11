import mmcv
import numpy as np
import os.path as osp
from tools.convert_datasets.voc_to_coco import CoCoData
import getpass


def main():
    username = getpass.getuser()

    xml_dir = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech_New/annotations-xml/')
    json_dir = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech_New/annotations-json/')
    txt_dir = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech_New/imageSets/')
    img_dir = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech_New/images')
    mmcv.mkdir_or_exist(json_dir)

    # all train images
    train_filelist = osp.join(txt_dir, 'train-all.txt')
    img_all_names = mmcv.list_from_file(train_filelist)
    xml_all_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_all_names]
    img_all_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_all_names]
    coco_train = CoCoData(xml_all_paths, img_all_paths, osp.join(json_dir, 'train-all.json'))
    coco_train.convert()

    # all test images
    test_filelist = osp.join(txt_dir, 'test-all.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_test_names]
    coco_test = CoCoData(xml_test_paths, img_test_paths, osp.join(json_dir, 'test-all.json'))
    coco_test.convert()


if __name__ == '__main__':
    main()
