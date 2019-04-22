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


path = "D:\\!Google_Learning\\076_Kaggle\\!Git_UP\\2019-2nd-ml-month-with-kakr\\datasets\\"

train = pd.read_csv(path+"train.csv", index_col=0)
test = pd.read_csv(path+"test.csv", index_col=0)

RANDOM_SEED = 631
np.random.seed(RANDOM_SEED)

price_raw = train['price']
train.drop('price', axis = 1, inplace=True)

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

    # type check
    # for i in dataset.columns:
    #     colty = dataset[i].dtype
    #     if not colty == 'int64' and not colty == 'float64':
    #         print(i.ljust(15),'column is a', str(dataset[i].dtype).ljust(8), 'type')
    return dataset

cleaned = clean_data(train)
cleaned_test = clean_data(test)



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


geoge_train = geogege(cleaned)
geoge_test = geogege(cleaned_test)







# pca = PCA(n_components=2)
# principalComponents_updated = pca.fit_transform(geoge_train)

# cleaned_pca = np.concatenate((cleaned, principalComponents_updated), axis=1)


def scaler_dummy(dataset):

  scaler_mm = MinMaxScaler() 
  scaler_ma = MaxAbsScaler()
  scaler_sd = StandardScaler()
  scaler_rb = RobustScaler()

  numerical = list(dataset.columns)
  data_transform_mm = pd.DataFrame(data = dataset)
  data_transform_ma = pd.DataFrame(data = dataset)
  data_transform_sd = pd.DataFrame(data = dataset)
  data_transform_rb = pd.DataFrame(data = dataset)


  scaler_mm.fit(dataset[numerical])
  scaler_ma.fit(dataset[numerical])
  scaler_sd.fit(dataset[numerical])
  scaler_rb.fit(dataset[numerical])


  data_transform_mm[numerical] = scaler_mm.transform(dataset[numerical])
  data_transform_ma[numerical] = scaler_ma.transform(dataset[numerical])
  data_transform_sd[numerical] = scaler_sd.transform(dataset[numerical])
  data_transform_rb[numerical] = scaler_rb.transform(dataset[numerical])


  ## get dummies

  features_final_mm = pd.get_dummies(data_transform_mm)
  features_final_ma = pd.get_dummies(data_transform_ma)
  features_final_sd = pd.get_dummies(data_transform_sd)
  features_final_rb = pd.get_dummies(data_transform_rb)

  return features_final_mm, features_final_ma, features_final_sd, features_final_rb

# mm, ma, sd, rb = scaler_dummy(cleaned_pca)

x_train, x_test, y_train, y_test = train_test_split(geoge_train, price_raw, test_size = 0.1, random_state=RANDOM_SEED)
train_columns = [col for col in x_train.columns if col not in ['id','price']]


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
        self.param['seed'] = kwargs.get('seed', 0)
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
        self.param['seed'] = kwargs.get('seed', 0)
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
        'subsample':1.0, 'tol':0.1, 'validation_fraction':0.1, 'verbose':0}

gbr_model = SklearnWrapper(GradientBoostingRegressor, params=gbr_param)


gbr_train, gbr_test, lasso_cv_score = get_oof(gbr_model, x_train[train_columns].fillna(-1), y_train, x_test[train_columns].fillna(-1), 
                            rmse, NFOLDS=5, kfold_random_state=RANDOM_SEED)



# est = GradientBoostingRegressor()

# est.fit(X_train, y_train)



# # pridict
# pred_est = est.predict(X_test)
# # pred_est = np.expm1(pred_est)
# get_para_gbr = est.get_params

# rmse_est_gbr = mean_squared_error(pred_est, y_test)
# print(rmse_est_gbr)







# def print_best_params(model, params):
#     grid_model = GridSearchCV(
#         model, 
#         param_grid = params,
#         cv=5,
#         scoring='neg_mean_squared_error')

#     grid_model.fit(X_train, y_train)
#     rmse = np.sqrt(-1*grid_model.best_score_)
#     rmse2 = mean_squared_error(pred_est, y_test)


#     print('{0} 5CV RMSE {1}, {2}, best alpha:{3}'.format(model.__class__.__name__, np.round(rmse, 4), grid_model.best_params_))
#     return grid_model.best_estimator_


# lgb_params7 = {'num_leaves': [10],
#          'min_data_in_leaf': [10], 
#          'objective':['regression'],
#          'max_depth': [-1],
#          'learning_rate': [0.05],
#          "min_child_samples": [10],
#          "boosting": ["gbdt"],
#          "feature_fraction": [0.9],
#          "bagging_freq": [1],
#          "bagging_fraction": [0.9] ,
#          "bagging_seed": [11],
#          "metric": ['rmse'],
#          "lambda_l1": [0.1],
#          "verbosity": [-1],
#          "nthread": [4],
#          'n_estimators':[5000],
#              'max_bin' : [100],
#          'refit':[True], 
#     'tree_method':['gpu_hist'],
#     'predictor':['gpu_predictor'],
#          "random_state": [RANDOM_SEED]}



# lgb_model = LGBMRegressor()

# lgb_estimator = print_best_params(lgb_model, lgb_params7)

# lgb_estimator = print_best_params(lgb_model, lgb_params7)










## 




# dtrain = xgb.DMatrix(X_train, y_train)
# dtest = xgb.DMatrix(X_test)

# xgb_params ={
#     'seed': [RANDOM_SEED],
#     'learning_rate': [0.02,0.03,0.04, 0.05],
#     'max_depth': [5,6],
#     'subsample': [0.8,0.9],
#     'colsample_bytree': [0.4,0.5],
#     'silent': [True],
#     'gpu_id':[0] ,         
#     'tree_method':['gpu_hist'],
#     'predictor':['gpu_predictor'],
#     'n_estimators':[1000],
#     'refit' : [True]
# }

# # cross validation
# cv_output = xgb.cv(xgb_params,
#                    dtrain,                        
#                    num_boost_round=15000,         # the number of boosting trees
#                    early_stopping_rounds=100,    # val loss가 계속 상승하면 중지
#                    nfold=5,                      # set folds of the closs validation
#                    verbose_eval=300,             # 몇 번째마다 메세지를 출력할 것인지
# #                    feval=rmse_exp,               # price 속성을 log scaling 했기 때문에, 다시 exponential
#                    maximize=False,
#                    show_stdv=False,              # 학습 동안 std(표준편차) 출력할지 말지
#                    )



# xgb_model = xgb.XGBRegressor() 

# xgb_estimator = print_best_params(xgb_model, xgb_params_jang)

