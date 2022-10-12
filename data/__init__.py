import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.voc_dataset import VOCDataSet
from data.voc_off_dataset_labeled import VOCLabelDataSet
from data.voc_off_dataset_unlabeled import VOCUnlabelDataSet
from data.new_city_loader import newcityscapesLoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "pascal_voc": VOCDataSet,
        "pascal_off_voc_labeled":VOCLabelDataSet,
        "pascal_off_voc_unlabeled":VOCUnlabelDataSet,
        "new_city":newcityscapesLoader
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return 'datasets/Cityscapes/data'
    if name == 'pascal_voc':
        return 'datasets/VOCdevkit/VOC2012'

    if name=='new_city':
        return 'dataset/rebuilt.leftImg8bit_trainextra333/'