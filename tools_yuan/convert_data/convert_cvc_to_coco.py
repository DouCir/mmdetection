import mmcv
import os.path as osp
from tools_yuan.convert_data.voc_to_coco import CoCoData

"""
Author: Yuan Yuan
Date:2018/12/16
Location:SCU
"""


def main():
    xml_dir = '/media/ser606/Data/DoubleCircle/CVC/annotations-xml/'
    json_dir = '/media/ser606/Data/DoubleCircle/CVC/annotations-json/'
    txt_dir = '/media/ser606/Data/DoubleCircle/CVC/imageSets/'
    img_dir = '/media/ser606/Data/DoubleCircle/CVC/images/'
    mmcv.mkdir_or_exist(json_dir)

    # all train images
    train_filelist = osp.join(txt_dir, 'train-all.txt')
    img_all_names = mmcv.list_from_file(train_filelist)
    img_all_names = [img_name.replace('annotations/', '') for img_name in img_all_names]
    xml_all_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_all_names]
    img_all_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_all_names]
    coco_train = CoCoData(xml_all_paths, img_all_paths, osp.join(json_dir, 'train-all.json'))
    coco_train.convert()

    # all test images
    test_filelist = osp.join(txt_dir, 'test-all.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    img_test_names = [img_name.replace('annotations/', '') for img_name in img_test_names]
    xml_test_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_test_names]
    coco_test = CoCoData(xml_test_paths, img_test_paths, osp.join(json_dir, 'test-all.json'))
    coco_test.convert()


if __name__ == '__main__':
    main()
