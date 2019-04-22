
# LGB

lgb_params1 = {
    'objective':['regression'],
    'num_leave' : [1],
    'learning_rate' : [0.05],
    'n_estimators':[1000],
    'max_bin' : [80],
    'gpu_id':[0] ,         
    'tree_method':['gpu_hist'],
    'predictor':['gpu_predictor'],
    'refit':[True],
    "random_state": [RANDOM_SEED]} # 장아저씨

lgb_params2 = {'num_leaves': [20,31,40],
         'min_data_in_leaf': [30], 
         'objective':['regression'],
         'max_depth': [-1],
         'learning_rate': [0.015,0.002,0.02,0.03,0.04,0.05],
         "min_child_samples": [20],
         "boosting": ["gbdt"],
         "feature_fraction":[ 0.9],
         "bagging_freq": [1],
         "bagging_fraction": [0.9] ,
         "bagging_seed": [11],
         "metric": ['rmse'],
         "lambda_l1": [0.1],
         "verbosity": [-1],
         "nthread": [4],
         "random_state": [RANDOM_SEED]} # 천성


lgb_params3 = {'num_leaves': [2,10,31],
         'min_data_in_leaf': [10,40], 
         'objective':['regression'],
         'max_depth': [-1],
         'learning_rate': [0.05],
         "min_child_samples": [10,30],
         "boosting": ["gbdt"],
         "feature_fraction": [0.9],
         "bagging_freq": [1],
         "bagging_fraction": [0.9] ,
         "bagging_seed": [11],
         "metric": ['rmse'],
         "lambda_l1": [0.1],
         "verbosity": [-1],
         "nthread": [4],
         'n_estimators':[1000],
             'max_bin' : [80,100],
         'refit':[True], 
    'tree_method':['gpu_hist'],
    'predictor':['gpu_predictor'],
         "random_state": [RANDOM_SEED]}


lgb_params4 = {'num_leaves': [10],
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
         'n_estimators':[5000,10000],
             'max_bin' : [100],
         'refit':[True], 
    'tree_method':['gpu_hist'],
    'predictor':['gpu_predictor'],
         "random_state": [RANDOM_SEED]}

lgb_params5 = {'num_leaves': [10],
         'min_data_in_leaf': [10], 
         'objective':['regression'],
         'max_depth': [-1],
         'learning_rate': [0.001,0.05],
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

# LGBMRegressor 5 CV �� ���� ��� RMSE �� 128756.4821, 
# ���� alpha:{'gpu_id': 0, 'learning_rate': 0.05, 'max_bin': 80, 'n_estimators': 1000, 'num_leave': 1, 'objective': 'regression', 'predictor': 'gpu_predictor', 'refit': True, 'tree_method': 'gpu_hist'

# LGBMRegressor 5 CV �� ���� ��� RMSE �� 177166.9585, ���� 
# alpha:{'bagging_fraction': 0.9, 'bagging_freq': 1, 'bagging_seed': 11, 'boosting': 'gbdt', 'feature_fraction': 0.9, 'lambda_l1': 0.1, 'learning_rate': 0.015, 'max_depth': -1, 'metric': 'rmse', 'min_child_samples': 20, 'min_data_in_leaf': 30, 'nthread': 4, 'num_leaves': 31, 'objective': 'regression', 'random_state': 631, 'verbosity': -1}

# LGBMRegressor 5 CV �� ���� ��� RMSE �� 128756.4821, ���� 
# alpha:{'gpu_id': 0, 'learning_rate': 0.05, 'max_bin': 80, 'n_estimators': 1000, 'num_leave': 1, 'objective': 'regression', 'predictor': 'gpu_predictor', 'random_state': 631, 'refit': True, 'tree_method': 'gpu_hist'}

# LGBMRegressor 5 CV �� ���� ��� RMSE �� 132617.7076, ���� 
# alpha:{'bagging_fraction': 0.9, 'bagging_freq': 1, 'bagging_seed': 11, 'boosting': 'gbdt', 'feature_fraction': 0.9, 'lambda_l1': 0.1, 'learning_rate': 0.05, 'max_depth': -1, 'metric': 'rmse', 'min_child_samples': 20, 'min_data_in_leaf': 30, 'nthread': 4, 'num_leaves': 40, 'objective': 'regression', 'random_state': 631, 'verbosity': -1}

# LGBMRegressor 5 CV �� ���� ��� RMSE �� 124160.1895, ���� 
# alpha:{'bagging_fraction': 0.9, 'bagging_freq': 1, 'bagging_seed': 11, 'boosting': 'gbdt', 'feature_fraction': 0.9, 'lambda_l1': 0.1, 'learning_rate': 0.05, 'max_bin': 80, 'max_depth': -1, 'metric': 'rmse', 'min_child_samples': 20, 'min_data_in_leaf': 30, 'n_estimators': 1000, 'nthread': 4, 'num_leaves': 10, 'objective': 'regression', 'predictor': 'gpu_predictor', 'random_state': 631, 'refit': True, 'tree_method': 'gpu_hist', 'verbosity': -1}

# LGBMRegressor 5 CV �� ���� ��� RMSE �� 121381.4537, ���� 
# alpha:{'bagging_fraction': 0.9, 'bagging_freq': 1, 'bagging_seed': 11, 'boosting': 'gbdt', 'feature_fraction': 0.9, 'lambda_l1': 0.1, 'learning_rate': 0.05, 'max_bin': 100, 'max_depth': -1, 'metric': 'rmse', 'min_child_samples': 10, 'min_data_in_leaf': 10, 'n_estimators': 1000, 'nthread': 4, 'num_leaves': 10, 'objective': 'regression', 'predictor': 'gpu_predictor', 'random_state': 631, 'refit': True, 'tree_method': 'gpu_hist', 'verbosity': -1}

# LGBMRegressor 5 CV �� ���� ��� RMSE �� 121381.4537, 
# ���� alpha:{'bagging_fraction': 0.9, 'bagging_freq': 1, 'bagging_seed': 11, 'boosting': 'gbdt', 'feature_fraction': 0.9, 'lambda_l1': 0.1, 'learning_rate': 0.05, 'max_bin': 100, 'max_depth': -1, 'metric': 'rmse', 'min_child_samples': 10, 'min_data_in_leaf': 10, 'n_estimators': 1000, 'nthread': 4, 'num_leaves': 10, 'objective': 'regression', 'predictor': 'gpu_predictor', 'random_state': 631, 'refit': True, 'tree_method': 'gpu_hist', 'verbosity': -1}

# LGBMRegressor 5 CV �� ���� ��� RMSE �� 119659.6849, ���� 
# alpha:{'bagging_fraction': 0.9, 'bagging_freq': 1, 'bagging_seed': 11, 'boosting': 'gbdt', 'feature_fraction': 0.9, 'lambda_l1': 0.1, 'learning_rate': 0.05, 'max_bin': 100, 'max_depth': -1, 'metric': 'rmse', 'min_child_samples': 10, 'min_data_in_leaf': 10, 'n_estimators': 5000, 'nthread': 4, 'num_leaves': 10, 'objective': 'regression', 'predictor': 'gpu_predictor', 'random_state': 631, 'refit': True, 'tree_method': 'gpu_hist', 'verbosity': -1}

# LGBMRegressor 5 CV �� ���� ��� RMSE �� 119659.6849, ���� 
# alpha:{'bagging_fraction': 0.9, 'bagging_freq': 1, 'bagging_seed': 11, 'boosting': 'gbdt', 'feature_fraction': 0.9, 'lambda_l1': 0.1, 'learning_rate': 0.05, 'max_bin': 100, 'max_depth': -1, 'metric': 'rmse', 'min_child_samples': 10, 'min_data_in_leaf': 10, 'n_estimators': 5000, 'nthread': 4, 'num_leaves': 10, 'objective': 'regression', 'predictor': 'gpu_predictor', 'random_state': 631, 'refit': True, 'tree_method': 'gpu_hist', 'verbosity': -1}
