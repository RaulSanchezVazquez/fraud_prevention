#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from geopy.distance import geodesic
from multiprocess import cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm

from fraud_prevention import config


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
    """
    """
    global DATA_GRP

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
        geolocations[:-1].values.tolist(),
        geolocations[1:].values.tolist(),
        geolocations.index[1:])

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

    return cc_transaction_features



def process(data):
    """
    """

    cc_numbers = data['credit_card_number'].value_counts().index
    DATA_GRP = data.groupby('credit_card_number')

    trasaction_features = apply_threading(
        func=process_cc_features,
        data=cc_numbers,
        verbose=True)

    trasaction_features = pd.concat(trasaction_features, axis=0)

