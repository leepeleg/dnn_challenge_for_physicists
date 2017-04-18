import numpy as np
import pandas as pd
import os

DATA_RESOLUTION_MIN = 15


class DFMeta(object):
    """
    A class to hold meta data about the different dataframes
    """
    def __init__(self, fname, ts_field_name):
        """
        :param fname: file name of the dataframe
        :param ts_field_name: column name for "timestamp" in this dataframe
        """
        self.fName = fname
        self.tsFieldName = ts_field_name


class DataSupplier(object):
    """
    Class for holding, loading and supplying relevant dataFrames.
    """
    def __init__(self, path2data):
        """ Initialize dataSupplier with the data in path2data
        :param path2data: path to the raw data frames
        """

        self.path = path2data

        self.dfs_meta = {
            'glucose': DFMeta('GlucoseValues.df', 'Timestamp'),
            'exercises': DFMeta('Exercises.df', 'Timestamp'),
            'meals': DFMeta('Meals.df', 'Timestamp'),
            'sleep': DFMeta('Sleep.df', 'sleep_time'),
            'test_food': DFMeta('TestFoods.df', 'Timestamp'),
            'bac': DFMeta('BacterialSpecies.df', None),
            'blood': DFMeta('BloodTests.df', None),
            'measurements': DFMeta('Measurements.df', None),
        }

        self.data = {}

        self.load_data(path2data)

    def load_data(self, path2data):
        """
        loads the data from path2data, currently supporting only one set of frames
        :param path2data: complete path to data
        """
        def resample(x):
            return x.reset_index(level=['ConnectionID'], drop=True). \
                resample(str(DATA_RESOLUTION_MIN) + 'min', label='right').last().ffill()

        for dfName, dfMeta in self.dfs_meta.items():
            fname = dfMeta.fName
            path2df = os.path.join(path2data, fname)
            self.data[dfName] = pd.read_pickle(path2df).sort_index()

        self.data['glucose'].index.names = ['ConnectionID', 'Timestamp']
        self.data['glucose'] = self.data['glucose'].sort_index().groupby(level='ConnectionID').apply(resample)

    def get_data(self, dfname, index=None):
        """
        returns a sub-dataFrame of requested kind for specified indices
        :param dfname: name of the data frame to return 
        :param index: specfic indices requested. If not specfied, return the entire table
        :return: dataFrame
        """
        # TODO: implement get_data function
        return self.data[dfname]
