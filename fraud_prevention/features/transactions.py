#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from geopy.distance import geodesic


def calculate_distance(loc1, loc2):
    """Calculate the distance between two geolocations.

    Parameters
    ----------
    loc1 : Tuple(float, float)
        The latitude and longitude of the first location.
    loc2 : Tuple(float, float)
        The latitude and longitude of the second location.

    Returns
    --------
    distance : float
        The distance in kilometers between the two points.

    Example
    -------
    ::

        # Example usage
        loc1 = (37.7749, -122.4194)  # San Francisco
        loc2 = (34.0522, -118.2437)  # Los Angeles

        distance = calculate_distance(loc1, loc2)
        print(f"The distance between the points is {distance:.2f} km")
    """
    return geodesic(loc1, loc2).kilometers

