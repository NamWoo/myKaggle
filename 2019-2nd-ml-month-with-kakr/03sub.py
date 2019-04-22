test = pd.read_csv("./gdrive/My Drive/!DataSets/2019-2nd-ml-month-with-kakr/test.csv", index_col=0)

def clean_data(dataset):
    # Explo data
    # print('Raw Dataset shape :'.ljust(36), 'col', dataset.shape[0], 'row', dataset.shape[1])
    null_list = {}
    for i in dataset.columns:
        colnull = dataset[i].isnull().sum()
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

cleaned = clean_data(test)

def PCPCA(dataset):
    pca = PCA(n_components=2)
    principalComponents_updated = pca.fit_transform(cleaned)
    return np.concatenate((cleaned, principalComponents_updated), axis=1)

cleaned_pca = PCPCA(cleaned)

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

mm, ma, sd, rb = scaler_dummy(cleaned_pca)

X_train, X_test, y_train, y_test = train_test_split(sd, price_raw, test_size = 0.1, random_state=RANDOM_SEED)


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
    'n_estimators':[1000,5000],
    'refit' : [True]
}
xgb_model = xgb.XGBRegressor() 

xgb_estimator = print_best_params(xgb_model, xgb_params_add1)


lgb_params6 = {'num_leaves': [10],
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


lgb_model = lgb.LGBMRegressor()
lgb_estimator = print_best_params(lgb_model, lgb_params6)


xgb_preds = xgb_estimator.predict(df_test)
lgb_preds = lgb_estimator.predict(df_test)
preds = 0.5* xgb_preds + 0.5*lgb_preds