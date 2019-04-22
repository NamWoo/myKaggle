xgb_model = xgb.XGBRegressor() 
xgb_estimator = print_best_params(xgb_model, xgb_params_jang)


xgb_params = {
    'eta': 0.02,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.4,
#     'tree_method': 'gpu_hist',    # 최적화된 분할 지점을 찾기 위한 algorithm 설정 + 캐글의 GPU 사용
#     'predictor': 'gpu_predictor', # 예측 시에도 GPU사용
    'objective': 'reg:linear',    # 회귀
    'eval_metric': 'rmse',        # kaggle에서 요구하는 검증모델
    'silent': True,               # 학습 동안 메세지 출력할지 말지
    'seed': RANDOM_SEED,

xgb_params_jang ={
    'seed': [RANDOM_SEED],
    'learning_rate': [0.02,0.03,0.04, 0.05],
    'max_depth': [5,6],
    'subsample': [0.8,0.9],
    'colsample_bytree': [0.4,0.5],
    'silent': [True],
    'gpu_id':[0] ,         
    'tree_method':['gpu_hist'],
    'predictor':['gpu_predictor'],
    'n_estimators':[1000],
    'refit' : [True]
}

# XGBRegressor 5 CV 시 최적 평균 RMSE 값 119135.0105, 최적 
# alpha:{'colsample_bytree': 0.4, 'gpu_id': 0, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 1000, 'predictor': 'gpu_predictor', 'refit': True, 'seed': 631, 'silent': True, 'subsample': 0.9, 'tree_method': 'gpu_hist'}


xgb_params_add1 ={
    'seed': [RANDOM_SEED],
    'learning_rate': [0.05],
    'max_depth': [5],
    'subsample': [0.9],
    'colsample_bytree': [0.4],
    'silent': [True],
    'gpu_id':[0] ,         
    'tree_method':['gpu_hist'],
    'predictor':['gpu_predictor'],
    'n_estimators':[5000],
    'refit' : [True]
}

xgb_params_add1 ={
      'seed': RANDOM_SEED,
      'learning_rate': 0.05,
      'max_depth': 5,
      'subsample': 0.9,
      'colsample_bytree': 0.4,
      'silent': True,
      'gpu_id':0 ,         
      'tree_method':'gpu_hist',
      'predictor':'gpu_predictor',
      'n_estimators':5000,
      'refit' : True
  }

xgb_model = xgb.XGBRegressor() 

xgb_estimator = print_best_params(xgb_model, xgb_params_add1)


# XGBRegressor 5 CV 시 최적 평균 RMSE 값 119056.2941, 최적 
# alpha:{'colsample_bytree': 0.4, 'gpu_id': 0, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 5000, 'predictor': 'gpu_predictor', 'refit': True, 'seed': 631, 'silent': True, 'subsample': 0.9, 'tree_method': 'gpu_hist'}


# XGBRegressor 5 CV 시 최적 평균 RMSE 값 121667.4475, 최적 
# alpha:{'colsample_bytree': 0.5, 'gpu_id': 0, 'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 1000, 'predictor': 'gpu_predictor', 'refit': True, 'silent': True, 'subsample': 0.8, 'tree_method': 'gpu_hist'}

# XGBRegressor 5 CV 시 최적 평균 RMSE 값 119135.0105, 최적 
# alpha:{'colsample_bytree': 0.4, 'gpu_id': 0, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 1000, 'predictor': 'gpu_predictor', 'refit': True, 'seed': 631, 'silent': True, 'subsample': 0.9, 'tree_method': 'gpu_hist'}



xgb_params_best = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.4,
#     'tree_method': 'gpu_hist',    # 최적화된 분할 지점을 찾기 위한 algorithm 설정 + 캐글의 GPU 사용
    'predictor': 'gpu_predictor', # 예측 시에도 GPU사용
    'objective': 'reg:linear',    # 회귀
    'eval_metric': 'rmse',        # kaggle에서 요구하는 검증모델
    'silent': True,               # 학습 동안 메세지 출력할지 말지
    'seed': RANDOM_SEED,
}


cv_output = xgb.cv(xgb_params_best,
                   dtrain,                        
                   num_boost_round=15000,         # the number of boosting trees
                   early_stopping_rounds=300,    # val loss가 계속 상승하면 중지
                   nfold=5,                      # set folds of the closs validation
                   verbose_eval=100,             # 몇 번째마다 메세지를 출력할 것인지
#                    feval=rmse_exp,               # price 속성을 log scaling 했기 때문에, 다시 exponential
                   maximize=False,
                   show_stdv=False,              # 학습 동안 std(표준편차) 출력할지 말지
                   )

# [5300]	train-rmse:34597.8	test-rmse:122531