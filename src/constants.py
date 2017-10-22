PROJECT_PATH = '~/github/UTSKaggle/'

DATA_DIR = 'data'
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
TRANSFORMED_INPUT_DIR = 'transformed_input'
MODEL_DIR = 'model'

DATA_PATH = PROJECT_PATH + DATA_DIR + '/'

DATA_PATH_INPUT = DATA_PATH + INPUT_DIR + '/'

DATA_PATH_OUTPUT = DATA_PATH + OUTPUT_DIR + '/'

DATA_PATH_TRANSFORMED_INPUT = DATA_PATH + TRANSFORMED_INPUT_DIR + '/'

DATA_PATH_MODEL = DATA_PATH + MODEL_DIR + '/'

INITIAL_COLUMNS = ['Age','Fnlwgt','Capital gain','Capital loss','Education years','Work hours per week',
                    'Occupation', 'Education level', 'Relationship status','Marital status','Employment class',
                    'Race','Sex','Native country','fw_bin']

TRAIN_COLUMNS = ['Age','Fnlwgt','Capital gain','Capital loss','Education years','Work hours per week',
                 'Occupation', 'Education level', 'Relationship status','Marital status','fw_bin']

TARGET_COLUMN = 'Salary'

cv_params = {'max_depth': [8,9,10],
             'min_child_weight': [1,3,5],
             'learning_rate': [0.3,0.35,0.4],
             'subsample': [0.85,0.9,0.95],
             'colsample_bytree': [0.85, 0.9, 0.95],
             'n_estimators': [350,400,450]
             }

ind_params = {'seed': 0, 'objective': 'binary:logistic'}