#!/usr/bin/env python3# -*- coding: utf-8 -*-from sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerfrom sklearn.linear_model import LogisticRegressionfrom sklearn.ensemble import RandomForestClassifierfrom xgboost import XGBClassifierfrom lightgbm import LGBMClassifierdef get_model_candidates():    """Get the candidate models to test.    Returns    --------    pipelines : dict[sklearn.pipeline.Pipeline]        The model pipelines.    param_grids : dict        The parameter grid.    """    # Model pipelines    pipelines = {        # 'random_forest': Pipeline([        #     ('model', RandomForestClassifier())        # ]),        # 'logistic_regression': Pipeline([        #     ('scaler', StandardScaler()),        #     ('model', LogisticRegression())        # ]),        'xgboost': Pipeline([            ('model', XGBClassifier(                tree_method='hist',                  n_estimators=1000,                use_label_encoder=False))        ]),        'lightgbm': Pipeline([            ('model', LGBMClassifier(n_estimators=1000))        ])    }    # Hyperparameter grids    param_grids = {        # 'random_forest': {        #     'model__n_estimators': [100, 200],        #     'model__max_depth': [10, 20]        # },        # 'logistic_regression': {        #     'model__C': [0.1, 1, 10],        #     'model__penalty': ['l1', 'l2'],        #     'model__solver': ['liblinear']        # },        'xgboost': {            # 'model__learning_rate': [0.01, 0.1, 0.2],            # 'model__subsample': [0.6, 0.8, 1.0],            # 'model__max_depth': [3, 6, 10],            # 'model__colsample_bytree': [0.6, 0.8, 1.0],            # 'model__reg_alpha': [0, 0.1, 1],            # 'model__reg_lambda': [1, 1.5, 2],            # 'model__min_child_weight': [1, 5, 10]            'model__colsample_bytree': [1.0],            'model__learning_rate': [0.1],            'model__max_depth': [3],            'model__min_child_weight': [5],            'model__reg_alpha': [1],            'model__reg_lambda': [2],            'model__subsample': [1.0]        },        'lightgbm': {            # 'model__n_estimators': [1000],            # 'model__learning_rate': [0.01, 0.05, 0.1],            # 'model__subsample': [0.4, 0.6, 0.8, 1.0],            # 'model__max_depth': [-1, 5, 10, 20],            # 'model__colsample_bytree': [0.6, 0.8, 1.0],            # 'model__reg_alpha': [0, 0.1, 1],            # 'model__reg_lambda': [0, 0.1, 1],            # 'model__min_child_samples': [10, 20, 50]            'model__colsample_bytree': [0.6],            'model__learning_rate': [0.05],            'model__max_depth': [10],            'model__min_child_samples': [50],            'model__n_estimators': [1000],            'model__reg_alpha': [1],            'model__reg_lambda': [0],            'model__subsample': [0.4]        }    }    return pipelines, param_grids