from partition_builder import pull_apart_validation_set, patients_with_more_than


def param_tuning(hfo_type_name, patients_dic):
    print('Analizying models for hfo type: {0} in {1}... '.format(hfo_type_name, 'Hippocampus'))
    patients_dic, _ = patients_with_more_than(0, hfo_type_name, patients_dic)
    model_patients, validation_patients = pull_apart_validation_set(patients_dic, hfo_type_name)
    model_patient_names = [p.id for p in model_patients]  # Obs mantiene el orden de model_patients
    field_names = ml_field_names(hfo_type_name)
    test_partition = get_balanced_partition(model_patients, hfo_type_name, K=4, method='balance_classes')
    column_names = []
    train_data = []
    labels = []
    partition_ranges = []
    i = 0
    for p_names in test_partition:
        test_patients = [patients_dic[name] for name in p_names]
        x, y = get_features_and_labels(test_patients, hfo_type_name, field_names)
        x_pd = pd.DataFrame(x)
        x_values = x_pd.values
        column_names = x_pd.columns
        scaler = RobustScaler()  # Scale features using statistics that are robust to outliers.
        x_values = scaler.fit_transform(x_values)
        analize_balance(y)
        x_values, y = balance_samples(x_values, y)
        train_data = train_data + list(x_values)
        labels = labels + list(y)
        partition_ranges.append((i, i + len(y)))
        i += len(y)

    data = pd.DataFrame(data=train_data, columns=column_names)
    data['soz'] = labels
    target = 'soz'
    predictors = column_names

    folds = [([i for i in range(len(data)) if (i < t_start or i >= t_end)],
              [i for i in range(t_start, t_end)])
             for t_start, t_end in partition_ranges]

    alg = XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=6,
        scale_pos_weight=1,
        seed=7)

    param_test1 = {
        'n_estimators': range(100, 200, 1000),
    }
    param_test2 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }

    # grid_search(alg, param_test, folds, fit_features=data[predictors].values, to_labels=data[target].values)

    param_configs = set_param_configs(param_test_2)
    config_result = {id: {'preds': [], 'probs': []} for id in param_configs.keys()}

    for id, c in param_configs.items():
        for train_idx, test_idx in folds:
            print('test_indexes {0}'.format(test_idx))
            train_features = data.iloc[train_idx].drop(columns=['soz']).values
            train_labels = data.iloc[train_idx]['soz']
            test_features = data.iloc[test_idx].drop(columns=['soz']).values
            test_labels = data.iloc[test_idx]['soz']
            alg.fit(train_features, train_labels, eval_metric='aucpr')
            test_predictions = alg.predict(test_features)
            test_probs = alg.predict_proba(test_features)[:, 1]
            config_result[id]['preds'] = config_result[id]['preds'] + list(test_predictions)
            config_result[id]['probs'] = config_result[id]['probs'] + list(test_probs)

    # Busco la config que tiene mejor metrica
    # Ver con Diego cual usar AP, f1score
    best_id = 1
    for id, result in config_result.items():  # probar si f1score da igual con probs y preds, ver si usa 0.5 simulando preds con ese thresh, si es eso podemos cambiar las preds segun un thresh
        average_precision = average_precision_score(labels, result['probs'])
        if average_precision > average_precision_score(labels, config_result[best_id]['probs']):
            best_id = id

def grid_search(alg, param_test, folds, fit_features, to_labels):
     gsearch = GridSearchCV(estimator=alg,
                            param_grid=param_test,
                            scoring='recall',
                            n_jobs=6,
                            iid=False,
                            cv=folds)
     gsearch.fit(fit_features, to_labels)

     print('GRID SEARCH RESULTS ')
     print(gsearch.cv_results_)
     print(gsearch.best_estimator_)
     print(gsearch.best_params_)
     print(gsearch.best_score_)

def get_param_configs(param_test):
     for k, r in param_test.items():
         param_test[k] = [i for i in r]

     i = 0
     permutations = [[]]
     for param_values in param_test.values():
         new_permutations = []
         for v in param_values:
             for p in permutations:
                 new_permutations.append(value)

     param_configs = {}
     id = 1
     for p in permutations:
         param_configs[id] = {list(param_test.keys())[i]: p[i] for i in range(len(list(param_test.keys())))}
         id += 1
     return param_configs

def set_param_config(alg, param_config):
     for feature in param_config.keys():
         alg.set_params(feature=param_config[feature])

'''
#No pude hacerla andar bien con el folds especficado, solo hace esas iteraciones e ignora el early stopping
def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=4, folds=None, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                          metrics='aucpr', early_stopping_rounds=early_stopping_rounds, folds=folds) #nfold=cv_folds
        print('Best fold has n_estimators: {0}'.format(cvresult.shape[0]))
        alg.set_params(n_estimators=cvresult.shape[0])
        return alg

'''
