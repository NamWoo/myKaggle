import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import lightgbm as lgb

from time import localtime
import time
from datetime import datetime, timedelta,date
import gc

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree  import DecisionTreeRegressor

from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_squared_error

# from plotnine import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import catboost as cb
from functools import wraps


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

pd.options.display.max_rows = 10000
pd.options.display.max_columns = 10000
pd.options.display.max_colwidth = 1000

RANDOM_SEED = 631
np.random.seed(RANDOM_SEED)

path = "D:\\!Google_Drive\\!DataSets\\re\\2019-2nd-ml-month-with-kakr\\"

train = pd.read_csv(path+"train.csv", index_col=0)
test = pd.read_csv(path+"test.csv", index_col=0)

price_raw = train['price']
price_raw_log = np.log1p(price_raw)

train.drop('price', axis = 1, inplace=True)


# type check
# date column is a object type
def type_checkk(dataset):
    for i in dataset.columns:
        colty = dataset[i].dtype
        if not colty == 'int64' and not colty == 'float64':
            print(i.ljust(15),'column is a', str(dataset[i].dtype).ljust(8), 'type')


def ch_model(X_train, y_train=price_raw):  
    xgb_params_add1 ={
        'seed': RANDOM_SEED,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.9,
        'colsample_bytree': 0.4,
        'silent': True,
        # 'gpu_id':0 ,         
        # 'tree_method':'gpu_hist',
        # 'predictor':'gpu_predictor',
        'n_estimators':5000,
        'refit' : True
    }

    ch_xgb_model = xgb.XGBRegressor() 
    #   x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=RANDOM_SEED)
    dtrain = xgb.DMatrix(X_train, y_train)
    # dtest = xgb.DMatrix(X_test)

    # cross validation
    cv_out = xgb.cv(xgb_params_add1,
                    dtrain,                        
                    num_boost_round=20000,         # the number of boosting trees
                    early_stopping_rounds=500,    # val loss가 계속 상승하면 중지
                    nfold=5,                      # set folds of the closs validation
                    verbose_eval=300,             # 몇 번째마다 메세지를 출력할 것인지
    #                    feval=rmse_exp,               # price 속성을 log scaling 했기 때문에, 다시 exponential
                    maximize=False,
                    show_stdv=False,              # 학습 동안 std(표준편차) 출력할지 말지
                    )

# [2700]	train-rmse:28041.2	test-rmse:116153
# ch_model(cleaned_train)
def clean_data(dataset):

    # Explo data
    # print('Raw Dataset shape :'.ljust(36), 'col', dataset.shape[0], 'row', dataset.shape[1])
    null_list = {}
    for i in train.columns:
        colnull = train[i].isnull().sum()
        if not colnull == 0:
            null_list[i] = colnull
    # print('missing value colnames and counts : '.ljust(36), null_list)

    # date column
    dataset['data_y'] = ''
    dataset['data_m'] = ''
    dataset['data_y'] = dataset['date'].apply(lambda x : str(x[:4])).astype(int)
    dataset['data_m'] = dataset['date'].apply(lambda x : str(x[4:6])).astype(int)
    dataset.drop('date', axis=1, inplace=True)
    # print('date dropped Dataset shape :'.ljust(36), 'col', dataset.shape[0], 'row', dataset.shape[1])

    return dataset


def geogege(data):
    data['zipcode'] = data['zipcode'].astype(str)  
    data['zipcode-3'] = data['zipcode'].apply(lambda x : str(x[2:3])).astype(int)
    data['zipcode-4'] = data['zipcode'].apply(lambda x : str(x[3:4])).astype(int)
    data['zipcode-5'] = data['zipcode'].apply(lambda x : str(x[4:5])).astype(int)
    data['zipcode-34'] = data['zipcode'].apply(lambda x : str(x[2:4])).astype(int)
    data['zipcode-45'] = data['zipcode'].apply(lambda x : str(x[3:5])).astype(int)
    data['zipcode-35'] = data['zipcode'].apply(lambda x : str(x[2:5])).astype(int)
    data.drop('zipcode', axis=1, inplace=True)
    return data


def ppcca2(trainset, testset):
    pca2 = PCA(n_components=2)
    coord = trainset[['lat','long']]
    coord_test = testset[['lat','long']]
    
    principalComponents_updated = pca2.fit_transform(coord)
    trainset['coord_pca1']= ''
    trainset['coord_pca2']= ''
    trainset['coord_pca1']= principalComponents_updated[:, 0]
    trainset['coord_pca2']= principalComponents_updated[:, 1]

    principalComponents_updated_test = pca2.transform(coord_test)
    testset['coord_pca1']= ''
    testset['coord_pca2']= ''
    testset['coord_pca1']= principalComponents_updated_test[:, 0]
    testset['coord_pca2']= principalComponents_updated_test[:, 1]
    return trainset, testset


def ppcca1(trainset, testset):
    pca1 = PCA(n_components=2)

    principalComponents_updated = pca1.fit_transform(trainset)
    trainset['pca1']= ''
    trainset['pca2']= ''
    trainset['pca1']= principalComponents_updated[:, 0]
    trainset['pca2']= principalComponents_updated[:, 1]

    principalComponents_updated_test = pca1.transform(testset)
    testset['pca1']= ''
    testset['pca2']= ''
    testset['pca1']= principalComponents_updated_test[:, 0]
    testset['pca2']= principalComponents_updated_test[:, 1]
    return trainset, testset


def Skewed_CF(dataset):
    skewed = ['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15', 'sqft_above', 'sqft_basement']
    features_log_transformed = pd.DataFrame(data = dataset)
    features_log_transformed[skewed] = dataset[skewed].apply(lambda x: np.log(x + 1))
    return features_log_transformed


def scaler_dummy(dataset,dataset_test):

    scaler_mm = MinMaxScaler() 
    scaler_ma = MaxAbsScaler()
    scaler_sd = StandardScaler()
    scaler_rb = RobustScaler()

    numerical = list(dataset.columns)
    data_transform_mm = pd.DataFrame(data = dataset)
    data_transform_ma = pd.DataFrame(data = dataset)
    data_transform_sd = pd.DataFrame(data = dataset)
    data_transform_rb = pd.DataFrame(data = dataset)

    data_transform_mm[numerical] = scaler_mm.fit_transform(dataset[numerical])
    data_transform_ma[numerical] = scaler_ma.fit_transform(dataset[numerical])
    data_transform_sd[numerical] = scaler_sd.fit_transform(dataset[numerical])
    data_transform_rb[numerical] = scaler_rb.fit_transform(dataset[numerical])
  #     scaler_mm.fit(dataset[numerical])
  #     scaler_ma.fit(dataset[numerical])
  #     scaler_sd.fit(dataset[numerical])
  #     scaler_rb.fit(dataset[numerical])

    data_transform_mm[numerical] = scaler_mm.transform(dataset[numerical])
    data_transform_ma[numerical] = scaler_ma.transform(dataset[numerical])
    data_transform_sd[numerical] = scaler_sd.transform(dataset[numerical])
    data_transform_rb[numerical] = scaler_rb.transform(dataset[numerical])

    ## get dummies
    features_final_mm = pd.get_dummies(data_transform_mm)
    features_final_ma = pd.get_dummies(data_transform_ma)
    features_final_sd = pd.get_dummies(data_transform_sd)
    features_final_rb = pd.get_dummies(data_transform_rb)

    numerical = list(dataset_test.columns)
    scaler_mm_fitted_test = scaler_mm.transform(dataset_test[numerical])
    scaler_ma_fitted_test = scaler_ma.transform(dataset_test[numerical])
    scaler_sd_fitted_test = scaler_sd.transform(dataset_test[numerical])
    scaler_rb_fitted_test = scaler_rb.transform(dataset_test[numerical])

    scaler_mm_fitted_test = pd.DataFrame(data = scaler_mm_fitted_test,columns=numerical)
    scaler_ma_fitted_test = pd.DataFrame(data = scaler_ma_fitted_test,columns=numerical)
    scaler_sd_fitted_test = pd.DataFrame(data = scaler_sd_fitted_test,columns=numerical)
    scaler_rb_fitted_test = pd.DataFrame(data = scaler_rb_fitted_test,columns=numerical)
    
    features_final_mmt = pd.get_dummies(scaler_mm_fitted_test)
    features_final_mat = pd.get_dummies(scaler_ma_fitted_test)
    features_final_sdt = pd.get_dummies(scaler_sd_fitted_test)
    features_final_rbt = pd.get_dummies(scaler_rb_fitted_test)        
    return features_final_mm, features_final_ma, features_final_sd, features_final_rb, features_final_mmt, features_final_mat, features_final_sdt, features_final_rbt
  

def get_oof(clf, x_train, y_train, x_test, eval_func, **kwargs):
    nfolds = kwargs.get('NFOLDS', 5)
    kfold_shuffle = kwargs.get('kfold_shuffle', True)
    kfold_random_state = kwargs.get('kfold_random_state', 0)
    stratified_kfold_ytrain = kwargs.get('stratifed_kfold_y_value', None)
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    
    kf_split = None
    if stratified_kfold_ytrain is None:
        kf = KFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)
        kf_split = kf.split(x_train)
    else:
        kf = StratifiedKFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)
        kf_split = kf.split(x_train, stratified_kfold_ytrain)
        
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))

    cv_sum = 0
    
    # before running model, print model param
    # lightgbm model and xgboost model use get_params()
    try:
        if clf.clf is not None:
            print(clf.clf)
    except:
        print(clf)
        print(clf.get_params())

    for i, (train_index, cross_index) in enumerate(kf_split):
        x_tr, x_cr = None, None
        y_tr, y_cr = None, None
        if isinstance(x_train, pd.DataFrame):
            x_tr, x_cr = x_train.iloc[train_index], x_train.iloc[cross_index]
            y_tr, y_cr = y_train.iloc[train_index], y_train.iloc[cross_index]
        else:
            x_tr, x_cr = x_train[train_index], x_train[cross_index]
            y_tr, y_cr = y_train[train_index], y_train[cross_index]

        clf.train(x_tr, y_tr, x_cr, y_cr)
        
        oof_train[cross_index] = clf.predict(x_cr)

        cv_score = eval_func(y_cr, oof_train[cross_index])
        
        print('Fold %d / ' % (i+1), 'CV-Score: %.6f' % cv_score)
        cv_sum = cv_sum + cv_score
        
        del x_tr, x_cr, y_tr, y_cr
        
    gc.collect()
    
    score = cv_sum / nfolds
    print("Average CV-Score: ", score)

    # Using All Dataset, retrain
    clf.train(x_train, y_train)
    oof_test = clf.predict(x_test)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), score


def time_decorator(func): 
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("\nStartTime: ", datetime.now() + timedelta(hours=9))
        start_time = time.time()
        
        df = func(*args, **kwargs)
        
        print("EndTime: ", datetime.now() + timedelta(hours=9))  
        print("TotalTime: ", time.time() - start_time)
        return df
        
    return wrapper


class SklearnWrapper(object):
    def __init__(self, clf, params=None, **kwargs):
        #if isinstance(SVR) is False:
        #    params['random_state'] = kwargs.get('seed', 0)
        self.clf = clf(**params)
        self.is_classification_problem = True
    @time_decorator
    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        if len(np.unique(y_train)) > 30:
            self.is_classification_problem = False
            
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        if self.is_classification_problem is True:
            return self.clf.predict_proba(x)[:,1]
        else:
            return self.clf.predict(x)   


class XgbWrapper(object):
    def __init__(self, params=None, **kwargs):
        self.param = params
        self.param['seed'] = kwargs.get('seed', RANDOM_SEED)
        self.num_rounds = kwargs.get('num_rounds', 1000)
        self.early_stopping = kwargs.get('ealry_stopping', 100)

        self.eval_function = kwargs.get('eval_function', None)
        self.verbose_eval = kwargs.get('verbose_eval', 100)
        self.best_round = 0
    
    @time_decorator
    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        need_cross_validation = True
       
        if isinstance(y_train, pd.DataFrame) is True:
            y_train = y_train[y_train.columns[0]]
            if y_cross is not None:
                y_cross = y_cross[y_cross.columns[0]]

        if x_cross is None:
            dtrain = xgb.DMatrix(x_train, label=y_train, silent= True)
            train_round = self.best_round
            if self.best_round == 0:
                train_round = self.num_rounds
            
            print(train_round)
            self.clf = xgb.train(self.param, dtrain, train_round)
            del dtrain
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train, silent=True)
            dvalid = xgb.DMatrix(x_cross, label=y_cross, silent=True)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

            self.clf = xgb.train(self.param, dtrain, self.num_rounds, watchlist, feval=self.eval_function,
                                 early_stopping_rounds=self.early_stopping,
                                 verbose_eval=self.verbose_eval)
            self.best_round = max(self.best_round, self.clf.best_iteration)

    def predict(self, x):
        return self.clf.predict(xgb.DMatrix(x), ntree_limit=self.best_round)

    def get_params(self):
        return self.param    


class LgbmWrapper(object):
    def __init__(self, params=None, **kwargs):
        self.param = params
        self.param['seed'] = kwargs.get('seed', RANDOM_SEED)
        self.num_rounds = kwargs.get('num_rounds', 1000)
        self.early_stopping = kwargs.get('ealry_stopping', 100)

        self.eval_function = kwargs.get('eval_function', None)
        self.verbose_eval = kwargs.get('verbose_eval', 100)
        self.best_round = 0
        
    @time_decorator
    def train(self, x_train, y_train, x_cross=None, y_cross=None):
        """
        x_cross or y_cross is None
        -> model train limted num_rounds
        
        x_cross and y_cross is Not None
        -> model train using validation set
        """
        if isinstance(y_train, pd.DataFrame) is True:
            y_train = y_train[y_train.columns[0]]
            if y_cross is not None:
                y_cross = y_cross[y_cross.columns[0]]

        if x_cross is None:
            dtrain = lgb.Dataset(x_train, label=y_train, silent= True)
            train_round = self.best_round
            if self.best_round == 0:
                train_round = self.num_rounds
                
            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=train_round)
            del dtrain   
        else:
            dtrain = lgb.Dataset(x_train, label=y_train, silent=True)
            dvalid = lgb.Dataset(x_cross, label=y_cross, silent=True)
            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=self.num_rounds, valid_sets=[dtrain, dvalid],
                                  feval=self.eval_function, early_stopping_rounds=self.early_stopping,
                                  verbose_eval=self.verbose_eval)
            self.best_round = max(self.best_round, self.clf.best_iteration)
            del dtrain, dvalid
            
        gc.collect()
    
    def predict(self, x):
        return self.clf.predict(x, num_iteration=self.clf.best_iteration)
    
    def plot_importance(self):
        lgb.plot_importance(self.clf, max_num_features=50, height=0.7, figsize=(10,30))
        plt.show()
        
    def get_params(self):
        return self.param


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))


gbr_param = {'alpha':0.9, 'criterion':'friedman_mse', 'init':None, 'learning_rate':0.1, 'loss':'ls', 'max_depth':5,
        'max_features':None, 'max_leaf_nodes':None,
        'min_impurity_decrease':0.0, 'min_impurity_split':None,
        'min_samples_leaf':1, 'min_samples_split':2,
        'min_weight_fraction_leaf':0.0, 'n_estimators':200,
        'n_iter_no_change':None, 'presort':'auto', 'random_state':RANDOM_SEED,
        'subsample':1.0, 'tol':0.1, 'validation_fraction':0.1, 'verbose':0
        }

xgb_params = {
    'eval_metric': 'rmse',
    'seed': RANDOM_SEED,
    'eta': 0.05,
    'gamma':0,
    'max_depth':5,
    'reg_alpha':0.00006,
    'subsample': 0.9,
    'colsample_bytree': 0.4,
    'silent': 1,
}

xgb_params1 = {
    'eval_metric': 'rmse',
    'seed': RANDOM_SEED,
    'eta': 0.0123,
    'gamma':0,
    'max_depth':3,
    'reg_alpha':0.00006,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'silent': 1,
        # 'seed': RANDOM_SEED,
        # 'learning_rate': 0.05,
        # 'max_depth': 5,
        # 'subsample': 0.9,
        # 'colsample_bytree': 0.4,
        # 'silent': True,
        # # 'gpu_id':0 ,         
        # # 'tree_method':'gpu_hist',
        # # 'predictor':'gpu_predictor',
        # 'n_estimators':5000,
        # 'refit' : True
}

lgb_params7 = {'num_leaves': 10,
         'min_data_in_leaf': 10, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.05,
         "min_child_samples": 10,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         'n_estimators':5000,
             'max_bin' : 100,
        #  'refit':True, 
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
         "random_state": RANDOM_SEED}




# type_checkk(train)
cleaned_train = clean_data(train)
cleaned_test = clean_data(test)

geoge_train = geogege(cleaned_train)
geoge_test = geogege(cleaned_test)

pcaed_train, pcaed_test = ppcca2(geoge_train, geoge_test)
pcaed_train1, pcaed_test1 = ppcca1(pcaed_train, pcaed_test)

# Skewed_train = Skewed_CF(pcaed_train1)
# Skewed_test = Skewed_CF(pcaed_test1)
# mm, ma, sd, rb, mmt, mat, sdt, rbt = scaler_dummy(Skewed_train, Skewed_test)


# mm, ma, sd, rb
# x_train, x_test, y_train, y_test = train_test_split(sd, price_raw_log, test_size = 0.1, random_state=RANDOM_SEED)

x_train = pcaed_train1
y_train = price_raw_log
x_test = pcaed_test1

gbr_model = SklearnWrapper(GradientBoostingRegressor, params=gbr_param)
xgb_model = XgbWrapper(params=xgb_params, num_rounds = 20000, ealry_stopping=500, verbose_eval=300)
lgb_model = LgbmWrapper(params=lgb_params7, num_rounds = 20000, ealry_stopping=500, verbose_eval=300)

train_columns = [col for col in x_train.columns if col not in ['id','price']]

xgb_train, xgb_test, xgb_cv_score = get_oof(xgb_model, x_train[train_columns], y_train, x_test[train_columns], 
                            rmse, NFOLDS=5, kfold_random_state=RANDOM_SEED)

lgb_train, lgb_test, lgb_cv_score = get_oof(lgb_model, x_train[train_columns], y_train, x_test[train_columns], 
                            rmse, NFOLDS=5, kfold_random_state=RANDOM_SEED)

gbr_train, gbr_test, lasso_cv_score = get_oof(gbr_model, x_train[train_columns].fillna(-1), y_train, x_test[train_columns].fillna(-1), 
                            rmse, NFOLDS=5, kfold_random_state=RANDOM_SEED)



# 마무리

x_train_second_layer = np.concatenate((lgb_train, xgb_train, gbr_train), axis=1)
x_test_second_layer = np.concatenate((lgb_test, xgb_test, gbr_test), axis=1)

x_train_second_layer = pd.DataFrame(x_train_second_layer)
x_test_second_layer = pd.DataFrame(x_test_second_layer)

x_train_second_layer.to_csv(path+'train_oof.csv', index=False)
x_test_second_layer.to_csv(path+'test_oof.csv', index=False)



xgb_ans = np.expm1(xgb_test)
lgb_ans = np.expm1(lgb_test)
gbr_ans = np.expm1(gbr_test)


en_ans22 = xgb_ans*0.6 + lgb_ans*0.2 + gbr_ans*0.2
en_ans11 = xgb_ans*0.8 + lgb_ans*0.1 + gbr_ans*0.1

submission = pd.read_csv("sample_submission.csv")


len(submission)


submission['price']= en_ans22
submission.to_csv(path+'en_ans22.csv',index=False)

submission['price']= en_ans11
submission.to_csv(path+'en_ans11.csv',index=False)

submission['price']= xgb_ans
submission.to_csv(path+'xgb_ans.csv',index=False)