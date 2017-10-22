import pandas as pd
from kaggle import functions
import constants
import numpy as np

train = pd.read_csv(constants.DATA_PATH_INPUT + 'train.csv')
test = pd.read_csv(constants.DATA_PATH_INPUT + 'test.csv')

functions.categorise(train)
functions.categorise(test)

train.is_copy = False
test.is_copy = False

# Let's bucketise final weight
train.loc[:,'fw_bin'] = (train.loc[:,'Fnlwgt'] / 20000).astype(np.int32)

test.loc[:,'fw_bin'] = (test.loc[:,'Fnlwgt'] / 20000).astype(np.int32)

train.to_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'train.csv')
test.to_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'test.csv')
