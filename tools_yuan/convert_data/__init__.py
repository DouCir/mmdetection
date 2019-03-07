"""
Author:Yuan Yuan
Date:2019/02/11
Description: This Python package is used to covert xml data format to pkl format or coco format.
             In this project, pkl format is used.
"""
from .voc_to_coco import CoCoData
from .utils import parse_xml, parse_xml_coder, track_progress_yuan

__all__ = ['CoCoData', 'parse_xml', 'parse_xml_coder', 'track_progress_yuan']
