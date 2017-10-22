import pandas as pd
import constants
import xgboost as xgb

train = pd.read_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'train.csv')

train_tst = train.pop(constants.TARGET_COLUMN).values

train_trn = train[constants.INITIAL_COLUMNS]

xgdmat = xgb.DMatrix(train_trn, train_tst)

our_params = {'eta': 0.35, 'seed':0, 'subsample': 0.95, 'colsample_bytree': 0.96,
            'objective': 'binary:logistic', 'max_depth':9, 'min_child_weight':1, 'silent': 1}

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 400)

importances = final_gb.get_fscore()

importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True, ascending = False)

print(importance_frame.head(50))