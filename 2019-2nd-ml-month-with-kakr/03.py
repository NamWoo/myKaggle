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

import xgboost as xgb
import lightgbm as lgb

from time import time, localtime

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


pca = PCA(n_components=18)
principalComponents = pca.fit_transform(cleaned)

variances=[val/100 for val in pca.explained_variance_]
cummilative_variances=np.cumsum(np.round(variances, decimals=3))

pca = PCA(n_components=2)
principalComponents_updated = pca.fit_transform(cleaned)

cleaned_pca = np.concatenate((cleaned, principalComponents_updated), axis=1)


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

mm, ma, sd, rb = scaler_dummy(cleaned)


X_train, X_test, y_train, y_test = train_test_split(sd, price_raw, test_size = 0.1, random_state=RANDOM_SEED)



start = time()
est = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
           learning_rate=0.1, loss='ls', max_depth=5,
           max_features=None, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=200,
           n_iter_no_change=None, presort='auto', random_state=RANDOM_SEED,
           subsample=1.0, tol=0.1, validation_fraction=0.1, verbose=0)

est.fit(X_train, y_train)

end = time()
time_d = end - start

# pridict
pred_est = est.predict(X_test)
# pred_est = np.expm1(pred_est)
get_para_gbr = est.get_params

rmse_est_gbr = mean_squared_error(pred_est, y_test)
# print(rmse_est_gbr)



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

def print_best_params(model, params):
    grid_model = GridSearchCV(
        model, 
        param_grid = params,
        cv=5,
        scoring='neg_mean_squared_error')

    grid_model.fit(X_train, y_train)
    rmse = np.sqrt(-1*grid_model.best_score_)
    print(
        '{0} 5 CV 시 최적 평균 RMSE 값 {1}, 최적 alpha:{2}'.format(model.__class__.__name__, np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_


lgb_params7 = {'num_leaves': [10],
         'min_data_in_leaf': [10], 
         'objective':['regression'],
         'max_depth': [-1],
         'learning_rate': [0.05],
         "min_child_samples": [10],
         "boosting": ["gbdt"],
         "feature_fraction": [0.9],
         "bagging_freq": [1],
         "bagging_fraction": [0.9] ,
         "bagging_seed": [11],
         "metric": ['rmse'],
         "lambda_l1": [0.1],
         "verbosity": [-1],
         "nthread": [4],
         'n_estimators':[5000],
             'max_bin' : [100],
         'refit':[True], 
    'tree_method':['gpu_hist'],
    'predictor':['gpu_predictor'],
         "random_state": [RANDOM_SEED]}



lgb_model = LGBMRegressor()
lgb_estimator = print_best_params(lgb_model, lgb_params7)

