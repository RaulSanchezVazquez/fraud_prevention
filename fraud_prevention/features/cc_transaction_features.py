#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocess import cpu_count
from geopy.distance import geodesic
from joblib import Parallel, delayed
from category_encoders.woe import WOEEncoder

from fraud_prevention import config
from fraud_prevention.features import creditcard


PATH = os.path.join(
    config.PRJ_DIR,
    'data/processed/cc_transaction_features.parquet')

# Aux. variable to async computation
DATA_GRP = None


def apply_threading(func, data, n_jobs=None, verbose=True):
    """Map a parallel function to a list using multithreading.

    Params:
    ---------
    func: function
        function The function to be pararellized
    data: list
        Items in which to apply func
    n_jobs: int
        number of threads
    verbose : bool
        Set to False prevents the display of the progress bar.

    Return:
    ---------
    list:
        Results in the same order as found in data

    Example:
    ---------

    #Simple example
    mylist = range(10)

    def power_of_two(x):
        return x**2

    apply_threading(
        power_of_two,
        mylist)

    Out:
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    """

    if n_jobs is None:
        n_jobs = cpu_count()

    if verbose:
        result = Parallel(
            n_jobs=n_jobs,
            backend='threading'
        )(
            delayed(func)(i) for i in tqdm(data)
        )
    else:
        result = Parallel(
            n_jobs=n_jobs,
            backend='threading'
        )(
            delayed(func)(i) for i in data
        )

    return result


def geo_distance_diff(geo1, geo2):
    """Calculate the distance between two geolocations.

    Distance is in kilometers scale.

    Parameters
    ----------
    geo1 : Tuple(float, float)
        The latitude and longitude of the first location.
    geo2 : Tuple(float, float)
        The latitude and longitude of the second location.

    Returns
    --------
    distance : float
        The distance in kilometers between the two points.

    Example
    -------
    ::

        from fraud_prevention.features import  feature_utils

        geo1 = (37.7749, -122.4194)  # San Francisco
        geo2 = (34.0522, -118.2437)  # Los Angeles

        feature_utils.geo_distance_diff(
            geo1=geo1,
            geo2=geo2)
        Out[1]: 559.0423365035714

    """
    return geodesic(geo1, geo2).kilometers


def process_cc_features(cc_number):
    """Process the features of a single credit card.

    Parameters
    ----------
    cc_number : int
        The credit card number.

    Return
    ------
    cc_transaction_features : pandas.DataFrame
        The data with the transaction features.

    """
    global DATA_GRP
    try:
        cc_data = DATA_GRP.get_group(cc_number)
        cc_data.sort_values('timestamp', inplace=True)

        geo_pairs_index = zip(
            cc_data[[
                'latitude',
                'longitude'
            ]][:-1].values.tolist(),
            cc_data[[
                'latitude',
                'longitude'
            ]][1:].values.tolist(),
            cc_data.index[1:])

        diff_geo = []
        for prev_geo, current_geo, index in geo_pairs_index:
            diff_geo.append({
                "index": index,
                "km_dist_prev_transaction": \
                    geo_distance_diff(geo1=prev_geo, geo2=current_geo)
            })

        time_pairs_index = zip(
            cc_data[:-1].values.tolist(),
            cc_data[1:].values.tolist(),
            cc_data.index[1:])

        time_pairs_index = zip(
            cc_data[
                'timestamp'
            ][:-1].values.tolist(),
            cc_data[
                'timestamp'
            ][1:].values.tolist(),
            cc_data.index[1:])

        diff_time = []
        for prev_time, current_time, index in time_pairs_index:
            diff_time.append({
                "index": index,
                "time_prev_transaction": (current_time - prev_time)
            })

        cc_transaction_features = pd.concat([
            pd.DataFrame(diff_time).set_index('index'),
            pd.DataFrame(diff_geo).set_index('index')
        ], axis=1)
    except:
        cc_transaction_features = pd.DataFrame()

    return cc_transaction_features


def get_merchant_charback_woe(data, window_size=500):
    """Get the merchant charback weight of evidence.

    The charback rate weight of evidence is computed per time window.

    To prevent the model to be fitted use the valid window.

    Parameters
    -----------
    data : pandas.DataFrame
        The data.
    window_size : int
        The time window size.

    Returns
    -------
    merchant_chargeback_woe : pandas.DataFrame
        The merchants chargeback weight of evidence at a give timestamp.
    """

    timestamps = data[
        'timestamp'
    ].apply(
        lambda x: x - (x % window_size)
    ).sort_values().drop_duplicates()

    merchant_chargeback_woe, pbar = [], tqdm(total=len(timestamps))
    for time in timestamps:
        time_data = data[data['timestamp'] < time]

        if time_data['Class'].sum() < 10:
            continue

        encoder = WOEEncoder(
            cols=['merchant']
        ).fit(
            time_data['merchant'],
            time_data['Class'])

        merchant_encoder = encoder.mapping['merchant']
        weights = pd.Series(
            merchant_encoder.values,
            index=encoder.ordinal_encoder.inverse_transform(
                pd.DataFrame(
                    {
                        "merchant": merchant_encoder.index
                    }
                )
            )['merchant'].tolist()
        ).sort_values().to_dict()
        weights['timestamp'] = time
        merchant_chargeback_woe.append(weights)
        pbar.update(1)

    merchant_chargeback_woe = pd.DataFrame(
        merchant_chargeback_woe
    ).set_index('timestamp')

    merchant_chargeback_woe = merchant_chargeback_woe.T
    merchant_chargeback_woe[None] = np.nan
    merchant_chargeback_woe = merchant_chargeback_woe.T

    return merchant_chargeback_woe


def process():
    """Process the credit card features.
    """
    global DATA_GRP

    data = creditcard.get()

    cc_numbers = data['credit_card_number'].value_counts().index
    DATA_GRP = data.groupby('credit_card_number')

    # Add transactional features
    trasaction_features = apply_threading(
        func=process_cc_features,
        data=cc_numbers,
        verbose=True)

    trasaction_features = pd.concat(trasaction_features, axis=0)
    trasaction_features.reset_index(inplace=True)

    dataset = data.reset_index().merge(
        trasaction_features,
        how='left',
        on=['index'])

    # Add temporal features
    merchant_chargeback_woe = get_merchant_charback_woe(
        data,
        window_size=500)

    # Get the valid WOE closest to the timestamp
    woe_valid_idx = []
    for t in dataset['timestamp']:
        woe_not_feature_leak = (
            merchant_chargeback_woe.index < t
        ) # Ensure not doing feature leak

        if woe_not_feature_leak.sum() == 0:
            woe_valid_idx.append(None)
        else:
            woe_valid_idx.append(
                merchant_chargeback_woe.index[
                    woe_not_feature_leak
                ].max()
            )
    dataset['merchant_chargeback_woe'] = [
        woe[m]
        for woe, m in zip(
            merchant_chargeback_woe.loc[
                pd.Series(woe_valid_idx)
            ].to_dict('records'),
            dataset['merchant'].tolist())]

    dataset.to_parquet(PATH)


def get():
    """Get the dataset.

    Returns
    --------
    data : pandas.DataFrame
        The data.
    """

    data = pd.read_parquet(PATH)

    data['latitude'] = data['latitude'].astype(float)
    data['longitude'] = data['longitude'].astype(float)

    return data
