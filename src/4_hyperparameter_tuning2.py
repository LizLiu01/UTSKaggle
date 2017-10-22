import pandas as pd
import constants
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

train = pd.read_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'train.csv')
test = pd.read_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'test.csv')

train_tst = train.pop(constants.TARGET_COLUMN).values

train_trn = train[constants.TRAIN_COLUMNS]

test_sub_trn = test[constants.TRAIN_COLUMNS]

optimized_GBM = GridSearchCV(xgb.XGBClassifier(**constants.ind_params), constants.cv_params, scoring = 'accuracy', cv = 2)

optimized_GBM.fit(train_trn, train_tst)

fe_res = optimized_GBM.cv_results_

for k in fe_res:
    if k == 'mean_train_score' or k == 'params' or k == 'mean_test_score':
        print(k)
        print(fe_res[k])