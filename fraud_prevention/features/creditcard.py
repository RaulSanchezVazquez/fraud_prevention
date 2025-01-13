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
] + (['MX'] * 5) + (['US'] * 5) + (['CA'] * 5)

FRAUDSTERS_TIME_DISTANCE_MIN = [0.5, 1, 1.5, 2, 2.5, 5, 10]
NON_FRAUDSTERS_TIME_DISTANCE_MIN = [10, 30, 60]


NON_FRAUDSTERS_MERCHANTS = [
    'gas_station_1',
    'gas_station_2',
    'restaurant_1',
    'restaurant_2',
    'store_1',
    'store_2',
    'fligh_tickets']

FRAUDSTERS_MERCHANTS = (
    NON_FRAUDSTERS_MERCHANTS
) + (['store_2'] * 10) + (['restaurant_2'] * 10) + (['fligh_tickets'] * 30)



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
        credit_card_number    346024495269014
        latitude                     41.72059
        longitude                   -87.70172
        timestamp                         0.0
        merchant                 restaurant_1
        V1                          -1.359807
        V2                          -0.072781
        V3                           2.536347
        V4                           1.378155
        V5                          -0.338321
        V6                           0.462388
        V7                           0.239599
        V8                           0.098698
        V9                           0.363787
        V10                          0.090794
        V11                           -0.5516
        V12                         -0.617801
        V13                          -0.99139
        V14                         -0.311169
        V15                          1.468177
        V16                         -0.470401
        V17                          0.207971
        V18                          0.025791
        V19                          0.403993
        V20                          0.251412
        V21                         -0.018307
        V22                          0.277838
        V23                         -0.110474
        V24                          0.066928
        V25                          0.128539
        V26                         -0.189115
        V27                          0.133558
        V28                         -0.021053
        Amount                         149.62
        Class                               0
        Name: 0, dtype: object

    """
    data = pd.read_parquet(PATH)

    return data


def process():
    """Add synthetic data.
    """
    data = creditcard.get()
    data_synthetic = get_synthetic_fraud(
        data,
        max_group_size=7)

    data = pd.concat([
        data_synthetic.reset_index(drop=True),
        data.drop(['Time'], axis=1).reset_index(drop=True)
    ], axis=1)

    data.to_parquet(PATH)


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
            group_size = random.choice(range(3, max_group_size + 1))

        # Create credit card number
        if local_group_size == 0:
            credit_card_number = FAKER.credit_card_number()

        # Create timestamps
        if is_fraud:
            timestamp_delta = random.choice(FRAUDSTERS_TIME_DISTANCE_MIN)
            country_code = random.choice(FRAUDSTERS_LOCATION)
            merchant = random.choice(FRAUDSTERS_MERCHANTS)
        else:
            timestamp_delta = random.choice(NON_FRAUDSTERS_TIME_DISTANCE_MIN)
            country_code = random.choice(NON_FRAUDSTERS_LOCATION)
            merchant = random.choice(NON_FRAUDSTERS_MERCHANTS)

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
            "timestamp": timestamp,
            "merchant": merchant
        })

        local_group_size += 1
        pbar.update(1)

    # Convert to pandas DataFrame
    dataset = pd.DataFrame(synthetic_data)

    return dataset
