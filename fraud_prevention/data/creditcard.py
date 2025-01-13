#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import requests
import pandas as pd

from fraud_prevention import config


def get():
    """Get the dataset.

    Returns
    --------
    data : pandas.DataFrame
        The data.

    Example
    -------
    ::

        from fraud_prevention.data import creditcard

        data = creditcard.get()

        data.head()
        Out[25]:
           Time        V1        V2        V3  ...       V28  Amount  Class
        0   0.0 -1.359807 -0.072781  2.536347  ...  0.133558  149.62      0
        1   0.0  1.191857  0.266151  0.166480  ... -0.008983  2.69        0
        2   1.0 -1.358354 -1.340163  1.773209  ... -0.055353  378.66      0
        3   1.0 -0.966272 -0.185226  1.792993  ...  0.062723  123.50      0
        4   2.0 -1.158233  0.877737  1.548718  ...  0.219422  69.99       0

        [5 rows x 31 columns]

        data.shape
    """
    data = pd.read_csv(
        os.path.join(config.PRJ_DIR, 'data/external/creditcard.csv')
    )

    return data
