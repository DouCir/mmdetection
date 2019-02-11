from .caltech import CaltechDataset
from .cvc import CvcDataset
from .kaist_single import KaistRGBDataset, KaistThermalDataset
from .kaist import KaistDataset

__all__ = ['CaltechDataset', 'CvcDataset', 'KaistDataset', 'KaistThermalDataset', 'KaistRGBDataset']
