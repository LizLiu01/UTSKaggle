import pandas as pd
import constants
import xgboost as xgb

train = pd.read_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'stacked_train.csv')
test = pd.read_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'test.csv')

train_tst = train.pop(constants.TARGET_COLUMN).values

train_trn = train[constants.TRAIN_COLUMNS]

test_sub_trn = test[constants.TRAIN_COLUMNS]

xgdmat = xgb.DMatrix(train_trn, train_tst)

our_params = {'eta': 0.35, 'seed':0, 'subsample': 0.95, 'colsample_bytree': 0.96,
            'objective': 'binary:logistic', 'max_depth':9, 'min_child_weight':1, 'silent': 1}

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 400)

save_path = '../' + constants.DATA_DIR + '/' + constants.MODEL_DIR + '/final_model.xgb'

final_gb.save_model(save_path)