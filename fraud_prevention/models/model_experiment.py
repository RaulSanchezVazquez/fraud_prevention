#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from fraud_prevention import config


X_PATH = os.path.join(
    config.PRJ_DIR,
    'models/X.parquet')

Y_PATH = os.path.join(
    config.PRJ_DIR,
    'models/y.parquet')

X_SHAP_VALUES_PATH = os.path.join(
    config.PRJ_DIR,
    'models/X_SHAP_VALUES_PATH.parquet')


def get_model_candidates():
    """Get the candidate models to test.

    Returns
    --------
    pipelines : dict[sklearn.pipeline.Pipeline]
        The model pipelines.
    param_grids : dict
        The parameter grid.

        'logistic_regression': Pipeline([
            ('Imputer', SimpleImputer(
                missing_values=np.nan,
                strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ]),
    """

    # Model pipelines
    pipelines = {
        'logistic_regression': Pipeline([
            ('Imputer', SimpleImputer(
                missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ]),
        'logistic_regression_SMOTE': ImbPipeline([
            ('Imputer', SimpleImputer(
                missing_values=np.nan, strategy='mean')),
            ('SMOTE', SMOTE(sampling_strategy='minority')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ]),
        'xgb': Pipeline([
            ('model', XGBClassifier(
                tree_method='hist', use_label_encoder=False))
        ]),
        'xgb_SMOTE': ImbPipeline([
            ('Imputer', SimpleImputer(
                missing_values=np.nan, strategy='constant', fill_value=0)),
            ('SMOTE', SMOTE(sampling_strategy='minority')),
            ('model', XGBClassifier(
                tree_method='hist', use_label_encoder=False))
        ]),
        'lightgbm': Pipeline([
            ('model', LGBMClassifier(
                verbose=-1, n_estimators=1000, num_leaves=100))
        ]),
        'lightgbm_SMOTE': ImbPipeline([
            ('Imputer', SimpleImputer(
                missing_values=np.nan, strategy='constant', fill_value=0)),
            ('SMOTE', SMOTE(sampling_strategy='minority')),
            ('model', LGBMClassifier(
                verbose=-1, n_estimators=1000, num_leaves=100))
        ])
    }

    # Hyperparameter grids
    param_grids = {
        'logistic_regression': {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear']
        },
        'logistic_regression_SMOTE': {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear']
        },
        'xgb': {
            'model__learning_rate': [0.01, 0.05],
            'model__max_depth': [5, 10],
            'model__colsample_bytree': [0.6],
            'model__n_estimators': [1000],
            'model__reg_alpha': [1],
            'model__reg_lambda': [0]
        },
        'xgb_SMOTE': {
            'model__learning_rate': [0.01, 0.05],
            'model__max_depth': [5, 10],
            'model__colsample_bytree': [0.6],
            'model__n_estimators': [1000],
            'model__reg_alpha': [1],
            'model__reg_lambda': [0]
        },
        'lightgbm': {
            'model__learning_rate': [0.01, 0.05],
            'model__max_depth': [5, 10],
            'model__colsample_bytree': [0.6],
            'model__min_child_samples': [50],
            'model__n_estimators': [1000],
            'model__reg_alpha': [1],
            'model__reg_lambda': [0]
        },
        'lightgbm_SMOTE': {
            'model__learning_rate': [0.01, 0.05],
            'model__max_depth': [5, 10],
            'model__colsample_bytree': [0.6],
            'model__min_child_samples': [50],
            'model__n_estimators': [1000],
            'model__reg_alpha': [1],
            'model__reg_lambda': [0]
        }
    }

    return pipelines, param_grids
