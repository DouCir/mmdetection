import os.path as osp
import mmcv
from tools.convert_datasets.utils import parse_xml
from tools.convert_datasets.utils import track_progress_yuan
import getpass

"""
author: Yuan Yuan
Date:2018/12/16
Location:SCU
"""


def main():
    username = getpass.getuser()

    xml_dir = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech_New/annotations-xml/')
    pkl_dir = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech_New/annotations-pkl/')
    txt_dir = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech_New/imageSets/')
    img_dir = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech_New/images')
    mmcv.mkdir_or_exist(pkl_dir)

    # all train images
    train_filelist = osp.join(txt_dir, 'train-all.txt')
    img_all_names = mmcv.list_from_file(train_filelist)
    xml_all_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_all_names]
    img_all_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_all_names]
    train_annotations = track_progress_yuan(parse_xml,
                                            list(zip(xml_all_paths, img_all_paths)))
    mmcv.dump(train_annotations, osp.join(pkl_dir, 'train-all.pkl'))

    # all test images
    test_filelist = osp.join(txt_dir, 'test-all.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_test_names]
    test_annotations = track_progress_yuan(parse_xml,
                                           list(zip(xml_test_paths, img_test_paths)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-all.pkl'))


if __name__ == '__main__':
    main()