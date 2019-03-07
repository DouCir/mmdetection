from .caltech import CaltechDataset
from .cvc import CvcDataset
from .kaist_single import KaistRGBDataset, KaistThermalDataset
from .kaist import KaistDataset
from .cvc_caltech import ExtendedCvcDataset
from .coder_dataset import CoderDataset
__all__ = ['CaltechDataset', 'CvcDataset', 'KaistDataset', 'KaistThermalDataset', 'KaistRGBDataset',
           'ExtendedCvcDataset', 'CoderDataset'
           ]
