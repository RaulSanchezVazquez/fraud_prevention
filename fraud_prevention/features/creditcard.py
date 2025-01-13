#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random

from tqdm import tqdm
import pandas as pd
import numpy as np
from faker import Faker

from fraud_prevention.data import creditcard
from fraud_prevention import config


PATH = os.path.join(
    config.PRJ_DIR,
    'data/processed/credit_card.parquet')

FAKER = Faker()

NON_FRAUDSTERS_LOCATION = [
    'MX',  # Mexico
    'US',  # USA
    'CA'  # Canada
]

FRAUDSTERS_LOCATION = [
    'AR',  # Argentina
    'DZ',  # Algeria
    'AT',  # Austria
    'AM',  # Armenia
    'AU',  # Australia
    'AZ',  # Azerbaijan
    'BD',  # Bangladesh
    'BE',  # Belgium
    'BR',  # Brazil
    'CL',  # Chile
    'CO',  # Colombia
    'CU',  # Cuba
    'EG',  # Egypt
] + (['MX'] * 10) + (['US'] * 10) + (['CA'] * 10)

FRAUDSTERS_TIME_DISTANCE_MIN = [0.5, 1, 5, 10]
NON_FRAUDSTERS_TIME_DISTANCE_MIN = [10, 30, 60]


def get():
    """Get the dataset.

    Returns
    --------
    data : pandas.DataFrame
        The data.

    Example
    -------
    ::

        from fraud_prevention.features import creditcard

        data = creditcard.get()
        data.iloc[0]
        Out[1]:

        redit_card_number    4116077007869887
        latitude                      40.49748
        longitude                      44.7662
        timestamp                      41181.0
        V1                           -7.334341
        V2                            4.960892
        V3                            -8.45141
        V4                            8.174825
        V5                           -7.237464
        V6                           -2.382711
        V7                          -11.508842
        V8                            4.635798
        V9                            -6.55776
        V10                         -11.519861
        V11                           6.455828
        V12                         -13.380222
        V13                           0.545279
        V14                         -13.026864
        V15                          -0.453595
        V16                         -13.251542
        V17                         -22.883999
        V18                          -9.287832
        V19                           4.038231
        V20                           0.723314
        V21                           2.153755
        V22                           0.033922
        V23                          -0.014095
        V24                            0.62525
        V25                           -0.05339
        V26                           0.164709
        V27                           1.411047
        V28                           0.315645
        Amount                           11.38
        Class                                1

    """
    if os.path.exists(PATH):
        data = pd.read_parquet(PATH)
    else:
        data = creditcard.get().sort_values('Class', ascending=False)
        data_synthetic = get_synthetic_fraud(
            data,
            max_group_size=7)

        data = pd.concat([
            data_synthetic.reset_index(drop=True),
            data.drop(['Time'], axis=1).reset_index(drop=True)
        ], axis=1)

        data.to_parquet(PATH)

    return data


def get_synthetic_fraud(data, max_group_size=7):
    """Function to generate synthetic dataset.

    The following synthetic data is added to the original data:
    - credit card number
    - latitude and longitude
    - timestamp


    Parameters
    -----------
    data : pandas.DataFrame
        The data.
    max_group_size : int
        The number of max transactions per credit card.

    Returns
    -------
    data : pandas.DataFrame
        The data.

    """

    synthetic_data = []
    local_group_size = 0

    group_size = random.choice(range(2, max_group_size + 1))

    pbar = tqdm(total=len(data))
    for _, row in data.iterrows():
        is_fraud = row['Class'] == 1
        if local_group_size >= group_size:
            local_group_size = 0
            group_size = random.choice(range(2, max_group_size + 1))

        # Create credit card number
        if local_group_size == 0:
            credit_card_number = FAKER.credit_card_number()

        # Create timestamps
        if is_fraud:
            timestamp_delta = random.choice(FRAUDSTERS_TIME_DISTANCE_MIN)
            country_code = random.choice(FRAUDSTERS_LOCATION)
        else:
            timestamp_delta = random.choice(NON_FRAUDSTERS_TIME_DISTANCE_MIN)
            country_code = random.choice(NON_FRAUDSTERS_LOCATION)

        if local_group_size == 0:
            timestamp = row['Time']
        else:
            timestamp = timestamp + timestamp_delta

        # Create geolocations
        lat, lon = FAKER.local_latlng(
            country_code=country_code,
            coords_only=True)

        synthetic_data.append({
            "credit_card_number": credit_card_number,
            "latitude": lat,
            "longitude": lon,
            "timestamp": timestamp
        })

        local_group_size += 1
        pbar.update(1)

    # Convert to pandas DataFrame
    dataset = pd.DataFrame(synthetic_data)

    return dataset
