import os.path as osp
import mmcv
from tools_yuan.convert_data.utils import parse_xml
from tools_yuan.convert_data.utils import track_progress_yuan
import getpass

"""
Author: Yuan Yuan
Date:2018/12/16
Location:SCU
"""


def main():
    username = getpass.getuser()

    pkl_dir = osp.join('/media/', username, 'Data/DoubleCircle/datasets/CVC-CaltechR/annotations-pkl/')
    mmcv.mkdir_or_exist(pkl_dir)

    xml_dir_caltech = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech/annotations-xml/')
    txt_dir_caltech = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech/imageSets/')
    img_dir_caltech = osp.join('/media/', username, 'Data/DoubleCircle/datasets/Caltech/images')

    xml_dir_cvc = osp.join('/media/', username, 'Data/DoubleCircle/datasets/CVC/annotations-xml/')
    txt_dir_cvc = osp.join('/media/', username, 'Data/DoubleCircle/datasets/CVC/imageSets/')
    img_dir_cvc = osp.join('/media/', username, 'Data/DoubleCircle/datasets/CVC/images')

    # caltech dataset(training)
    # all train images
    train_filelist = osp.join(txt_dir_caltech, 'train-all.txt')
    img_all_names = mmcv.list_from_file(train_filelist)
    xml_all_paths_caltech = [osp.join(xml_dir_caltech, img_name.replace('.txt', '.xml')) for img_name in img_all_names]
    img_all_paths_caltech = [osp.join(img_dir_caltech, img_name.replace('.txt', '.jpg')) for img_name in img_all_names]
    flags_caltech = ['train' for _ in img_all_names]

    # cvc dataset(training)
    # all train images
    train_filelist = osp.join(txt_dir_cvc, 'train-all.txt')
    img_all_names = mmcv.list_from_file(train_filelist)
    xml_all_paths_cvc = [osp.join(xml_dir_cvc, '{}.xml'.format(img_name)) for img_name in img_all_names]
    img_all_paths_cvc = [osp.join(img_dir_cvc, '{}.png'.format(img_name)) for img_name in img_all_names]
    flags_cvc = ['train' for _ in img_all_names]

    xml_all_paths = xml_all_paths_caltech + xml_all_paths_cvc
    img_all_paths = img_all_paths_caltech + img_all_paths_cvc
    flags = flags_caltech + flags_cvc

    train_annotations = track_progress_yuan(parse_xml,
                                            list(zip(xml_all_paths, img_all_paths, flags)))
    mmcv.dump(train_annotations, osp.join(pkl_dir, 'train-all.pkl'))

    # cvc dataset(testing)
    # all test images
    test_filelist = osp.join(txt_dir_cvc, 'test-all.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths_cvc = [osp.join(xml_dir_cvc, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths_cvc = [osp.join(img_dir_cvc, '{}.png'.format(img_name)) for img_name in img_test_names]
    flags_cvc = ['test' for _ in img_test_names]

    test_annotations = track_progress_yuan(parse_xml,
                                           list(zip(xml_test_paths_cvc, img_test_paths_cvc, flags_cvc)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-all.pkl'))


if __name__ == '__main__':
    main()
