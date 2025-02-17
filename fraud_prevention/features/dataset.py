#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_prevention.features import cc_transaction_features


def remove_neg_class_outliers(data, features):
    """Remove outliers in the negative class.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    features : list[str]
        The features.

    Returns
    -------
    data : pandas.DataFrame
        The data with outliers removed.
    """
    # Remove outliers
    is_f_outlier_cnt = []
    for f in features:
        min_val, max_val = data[f].quantile([
            .01,
            .99
        ]).sort_index().tolist()

        is_f_outlier = (
            (
                data[f] < min_val
            ) | (
                data[f] > max_val
            )
        ) & (
            data['Class'] == 0
        )

        is_f_outlier_cnt.append(is_f_outlier)

    is_f_outlier_cnt = pd.DataFrame(is_f_outlier_cnt).T.sum(axis=1)
    is_not_outlier = (is_f_outlier_cnt == 0)

    data = data[is_not_outlier]

    return data


def get(val_size=.1, test_size=0.3):
    """Get the dataset.

    Returns
    --------
    data : pandas.DataFrame
        The data.

    Examples
    ---------
    ::

        import pandas as pd
        from fraud_prevention.features import dataset

        test_size = 0.3
        val_size = .1

        (
            X_train, X_test, X_val,
            y_train, y_test, y_val,
            w_train, w_test, w_val
        ) = dataset.get(
            test_size=test_size,
            val_size=val_size)

        pd.DataFrame([
            {
                "name": "train",
                "size": X_train.shape[0],
                "pos_class": (y_train == 1).sum(),
                "new_class": (y_train == 0).sum()
            },
            {
                "name": "val",
                "size": X_val.shape[0],
                "pos_class": (y_val == 1).sum(),
                "new_class": (y_val == 0).sum()
            },
            {
                "name": "test",
                "size": X_test.shape[0],
                "pos_class": (y_test == 1).sum(),
                "new_class": (y_test == 0).sum()
            },
        ]).set_index('name')

        Out[1]:
                 size  pos_class  new_class
        name
        train    2435        218       2217
        val       271         28        243
        test   142404        246     142158

        # Not using SMOTE
        test_size = 0.3
        val_size = .1
        use_smote = False

        (
            X_train, X_test, X_val,
            y_train, y_test, y_val,
            w_train, w_test, w_val
        ) = dataset.get(
            test_size=test_size,
            val_size=val_size,
            use_smote=use_smote)

        pd.DataFrame([
            {
                "name": "train",
                "size": X_train.shape[0],
                "pos_class": (y_train == 1).sum(),
                "new_class": (y_train == 0).sum()
            },
            {
                "name": "val",
                "size": X_val.shape[0],
                "pos_class": (y_val == 1).sum(),
                "new_class": (y_val == 0).sum()
            },
            {
                "name": "test",
                "size": X_test.shape[0],
                "pos_class": (y_test == 1).sum(),
                "new_class": (y_test == 0).sum()
            },
        ]).set_index('name')
        Out[2]:
                size  pos_class  new_class
        name
        train  126384        312     126072
        val     14043         39      14004
        test    85443        141      85302
    """

    data = cc_transaction_features.get()
    data['is_known_merchant'] = data['is_known_merchant'].astype(float)

    features = [
        x
        for x in data.columns
        if x.startswith('V')
    ] + [
        'Amount',
        'time_prev_transaction',
        'km_dist_prev_transaction',
        'merchant_chargeback_woe',
        'is_known_merchant'
    ]

    # Get the test partition using an out-of-time strategy
    is_test = (
        data['timestamp'] > data['timestamp'].quantile(1 - test_size)
    )
    train_data, test_data = data[~is_test], data[is_test]

    # Remove outliers from the negative class
    # Negative class is aboundant so we can afford removing the outliners
    train_data = remove_neg_class_outliers(
        data=train_data,
        features=features)

    X_train, y_train, w_train = \
        train_data[features], train_data['Class'], train_data['Amount']

    X_test, y_test, w_test = \
        test_data[features], test_data['Class'], test_data['Amount']

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_train,
        y_train,
        w_train,
        test_size=val_size,
        random_state=42)

    return (
        X_train, X_test, X_val,
        y_train, y_test, y_val,
        w_train, w_test, w_val)
