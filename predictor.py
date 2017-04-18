import numpy as np
import pandas as pd
import os
from dataSupplier import DataSupplier

# The time series that you would get are such that the difference
# between two rows is 15 minutes. This is a global number that we
# used to prepare the data, so you would need it for different purposes
DATA_RESOLUTION_MIN = 15


class Predictor(object):
    """
    This is where you should implement your predictor.
    The testing script calls the 'predict' function with X.
    You should implement this function as you wish, just not the function signature.
    The other functions are here just as an example for you to have something to start with.
    """

    def __init__(self, path2data):
        """ Initialize predictor on the data in path2data
        :param path2data: path to the raw data frames
        """

        self.path = path2data

        self.X = None

        self.dataSupplier = DataSupplier(path2data)

    def predict(self, X):
        """ Given dataFrame of connection id and time stamp (X) predict
        the glucose level of each connection id 2 hours after timestamp

        :param X: Empty Multiindex dataframe with indexes ConnectionID and Timestamp
        :return: numpy array with the predictions for each row in X
                 (which is the following number for each row: glucose[timestamp+1hour] - glucose[timestamp])
        """

        # build features for set of (connID, timestamp)
        x = self.build_features(X)

        # feed the network
        y = x.sum(1)

        return y


    def build_features(self, X):
        """ Enhance the given table of X=(connectionIDs, timestamps) to a have more relevant data.
        Eventually it would be better to have a big table with lots of features for each row(connectionID, timestamp)
        :param X: A pandas DataFrame of rows of the form (connectionID, timestamp)
        :return: Enhanced dataframe with features
        """

        df1 = self.add_rolling_lags(X, self.dfs['glucose'],
                                    self.dfs['glucose'].columns,
                                    ['30min', '1h', '4h', '12h'],
                                    np.mean, 'mean')

        # for each (connectionID,timestamp), add some blood related data
        df2 = df1.join(self.dfs['blood'][['ALT', 'Calcium']]).fillna(0)

        return df2.values

    def add_rolling_lags(self, X, features_df, cols, lags, agg_func, agg_name):
        """ An example of an enhancing rolling function
        :param features_df: the dataframe that would be used for enhancing X
        :param cols: which cols we want to use for the enhancement
        :param lags: which lags to use for the rolling window
        :param agg_func: what aggregation function to use on each window
        :param agg_name: a string to be used for the resulting feature column
        :return: the original X dataframe together with the feature columns
        """
        # concat the features values to X to get a bigger table with all of them together
        df = pd.concat([X, features_df[cols]], axis=1)

        # for each (connectionID,timestamp) and for each lag, aggregate according to agg_func
        features = []
        for lag in lags:
            roll_lag = lambda group: group.reset_index(level=['ConnectionID'], drop=True).rolling(lag).apply(agg_func)
            f = df.fillna(0).sort_index().groupby(level='ConnectionID').apply(roll_lag)

            f.columns = ['{}_{}_{}'.format(col, agg_name, lag) for col in f.columns]
            features.append(f)
        # concatenate all feature vectors together to form on big enhanced dataframe
        df = pd.concat(features, 1)
        # keep only the rows from the original X
        df = df.loc[X.index]
        return df


if __name__ == "__main__":
    # example of Predict() usage

    # create Predictor instance
    path2data = '/net/mraid11/export/data/hadargo/dnn_challenge_train_data/'
    predict_inst = Predictor(path2data)

    # load x_y table : (connId, timestamp)---> label
    x_y = pd.read_pickle(os.path.join(path2data,'x_y.df'))

    # split the table to X (connId, timestamp) and Y (label)
    X = x_y.drop('label', axis=1)
    Y = np.asarray(x_y['label'].tolist())

    # predict Y
    y_pred = predict_inst.predict(X)

    # test the prediction
    score = np.mean(np.abs(y_pred - Y))

    print("Your score is: {}".format(score))

