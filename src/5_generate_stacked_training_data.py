import pandas as pd
import constants
import xgboost as xgb

train = pd.read_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'train.csv')
test = pd.read_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'test.csv')

train_tst = train.pop(constants.TARGET_COLUMN).values

train_trn = train[constants.TRAIN_COLUMNS]

test_sub_trn = test[constants.TRAIN_COLUMNS]

xgdmat = xgb.DMatrix(train_trn, train_tst)

our_params = {'eta': 0.35, 'seed':0, 'subsample': 0.95, 'colsample_bytree': 0.96,
            'objective': 'binary:logistic', 'max_depth':9, 'min_child_weight':1, 'silent': 1}

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 400)

test_new_dmat_sub = xgb.DMatrix(test_sub_trn)

y_pred = final_gb.predict(test_new_dmat_sub)

salarycol1 = pd.Series(y_pred, name='Salary')
salarycol2 = pd.Series(train_tst, name='Salary')

stest_sub = test_sub_trn.join(salarycol1)
strain_sub = train_trn.join(salarycol2)

stacked_train = pd.concat([strain_sub, stest_sub])

dtrain = xgb.DMatrix(stacked_train)

save_path = '../' + constants.DATA_DIR + '/' + constants.TRANSFORMED_INPUT_DIR + '/train.buffer'

dtrain.save_binary(save_path)