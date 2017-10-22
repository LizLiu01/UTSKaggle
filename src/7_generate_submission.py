import pandas as pd
import constants
import xgboost as xgb

test = pd.read_csv(constants.DATA_PATH_TRANSFORMED_INPUT + 'test.csv')

save_path = '../' + constants.DATA_DIR + '/' + constants.MODEL_DIR + '/final_model.xgb'

final_gb = xgb.Booster({'nthread': 4})

final_gb.load_model(save_path)

test_sub_trn = test[constants.TRAIN_COLUMNS]

test_new_dmat_sub = xgb.DMatrix(test_sub_trn)
y_pred = final_gb.predict(test_new_dmat_sub) # Predict using our test_new_dmat_sub

y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

y_results = map(int, y_pred)

csv_data = zip(test['ID'], y_results)

labels = ['ID','Salary']

final_df = pd.DataFrame.from_records(csv_data, columns=labels)

final_df.to_csv(constants.DATA_PATH_OUTPUT + 'xgb_output.csv',index = False)