import constants
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_tree

save_path = '../' + constants.DATA_DIR + '/' + constants.MODEL_DIR + '/final_model.xgb'

final_gb = xgb.Booster({'nthread': 4})

final_gb.load_model(save_path)

plot_tree(final_gb)
plt.show()