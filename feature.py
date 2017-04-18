from dataSupplier import DataSupplier
import numpy as np


class Feature(object):
    """An abstract class for sub-classing different features"""

    def __init__(self, data_supplier):
        if isinstance(data_supplier, DataSupplier):
            self.dataSupplier = data_supplier
        else:
            raise AttributeError('data_supplier must be of DataSupplier class')

    def compute(self, index, params=None):
        return np.zeros(len(index))


class FeatureMeanValue(Feature):

    def __init__(self, data_supplier):
        super().__init__(data_supplier)
        self.name = "MeanValue"

    def compute(self, index, params=None):
        data = self.dataSupplier.get_data('blood', index)
        return np.zeros(len(index))
