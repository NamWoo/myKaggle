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